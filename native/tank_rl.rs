//! 原版 SWF 的宿主接口。只控制输入、完整帧推进和读取状态，不实现游戏物理。
use crate::avm1::{Activation, ActivationIdentifier, Object, Value};
use crate::avm1::ExecutionReason;
use crate::backend::locale::DeterministicLocaleBackend;
use crate::backend::navigator::{NullExecutor, NullNavigatorBackend};
use crate::display_object::TDisplayObject;
use crate::events::{KeyDescriptor, KeyLocation, LogicalKey, NamedKey, PhysicalKey, PlayerEvent};
use crate::limits::ExecutionLimit;
use crate::string::AvmString;
use crate::{FloatDuration, LoadBehavior, Player, PlayerBuilder};
use chrono::TimeZone;
use serde_json::{json, Map, Value as Json};
use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::cell::{Cell, RefCell};
use crate::avm1::{FunctionObject, Attribute};

thread_local! {
    static HITS: RefCell<Vec<Json>> = const { RefCell::new(Vec::new()) };
    static HOST_FRAME: Cell<u64> = const { Cell::new(0) };
    static FIRES: RefCell<Vec<Json>> = const { RefCell::new(Vec::new()) };
}

fn observed_fire<'gc>(act: &mut Activation<'_, 'gc>, this: Object<'gc>, args: &[Value<'gc>]) -> Result<Value<'gc>, crate::avm1::Error<'gc>> {
    let original = prop(this, "__tank_rl_original_fire", act).as_object(act).expect("original fire function");
    let result = original.call("fireWeapon", act, this.into(), args)?;
    if args.len() >= 2 {
        if let Some(owner) = args[0].as_object(act) {
            let event = json!({"frame": HOST_FRAME.with(Cell::get),
                "owner": scalar(prop(owner, "username", act)), "weapon": scalar(args[1])});
            FIRES.with(|events| events.borrow_mut().push(event));
        }
    }
    Ok(result)
}

fn observed_hit<'gc>(act: &mut Activation<'_, 'gc>, this: Object<'gc>, args: &[Value<'gc>]) -> Result<Value<'gc>, crate::avm1::Error<'gc>> {
    let original = prop(this, "__tank_rl_original_hit", act).as_object(act).expect("original hit function");
    let result = original.call("registerHit", act, this.into(), args)?;
    if args.len() >= 2 {
        let owner = args[0].as_object(act);
        let victim = args[1].as_object(act);
        if let (Some(owner), Some(victim)) = (owner, victim) {
            let event = json!({"frame": HOST_FRAME.with(Cell::get),
                "owner": scalar(prop(owner, "username", act)), "victim": scalar(prop(victim, "username", act))});
            HITS.with(|events| events.borrow_mut().push(event));
        }
    }
    Ok(result)
}

fn install_hit_observer(player: &mut Player) {
    player.mutate_with_update_context(|ctx| {
        let clip = ctx.stage.root_clip().unwrap();
        let root = clip.object1().unwrap();
        let mut act = Activation::from_nothing(ctx, ActivationIdentifier::root("hit-observer"), clip);
        let original = prop(root, "registerHit", &mut act);
        let name = AvmString::new_utf8(act.gc(), "__tank_rl_original_hit");
        root.define_value(act.gc(), name, original, Attribute::DONT_ENUM);
        let proto = act.resolve_prototype([AvmString::new_utf8(act.gc(), "Function")]);
        let wrapper = FunctionObject::native(observed_hit).build(&act.context.strings, proto, None);
        let name = AvmString::new_utf8(act.gc(), "registerHit");
        root.define_value(act.gc(), name, wrapper.into(), Attribute::empty());
        let original = prop(root, "fireWeapon", &mut act);
        let name = AvmString::new_utf8(act.gc(), "__tank_rl_original_fire");
        root.define_value(act.gc(), name, original, Attribute::DONT_ENUM);
        let proto = act.resolve_prototype([AvmString::new_utf8(act.gc(), "Function")]);
        let wrapper = FunctionObject::native(observed_fire).build(&act.context.strings, proto, None);
        let name = AvmString::new_utf8(act.gc(), "fireWeapon");
        root.define_value(act.gc(), name, wrapper.into(), Attribute::empty());
    });
}

fn prop<'gc>(object: Object<'gc>, name: &str, act: &mut Activation<'_, 'gc>) -> Value<'gc> {
    let name = AvmString::new_utf8(act.gc(), name);
    object.get(name, act).unwrap_or(Value::Undefined)
}

fn sub<'gc>(object: Object<'gc>, name: &str, act: &mut Activation<'_, 'gc>) -> Option<Object<'gc>> {
    prop(object, name, act).as_object(act)
}

fn scalar(value: Value<'_>) -> Json {
    match value {
        Value::Bool(v) => json!(v),
        Value::Number(v) if v.is_finite() => json!(v),
        Value::String(v) => json!(v.to_utf8_lossy()),
        _ => Json::Null,
    }
}

fn fields<'gc>(object: Object<'gc>, names: &[&str], act: &mut Activation<'_, 'gc>) -> Json {
    let mut out = Map::new();
    for name in names {
        out.insert((*name).to_string(), scalar(prop(object, name, act)));
    }
    Json::Object(out)
}

fn dump_scalars<'gc>(object: Object<'gc>, act: &mut Activation<'_, 'gc>) -> Json {
    let mut out = Map::new();
    for (name, value) in object.own_properties() {
        let value = scalar(value);
        if !value.is_null() {
            out.insert(name.to_utf8_lossy().into_owned(), value);
        }
    }
    for name in ["_x", "_y", "_rotation", "_width", "_height", "_currentframe"] {
        out.insert(name.to_string(), scalar(prop(object, name, act)));
    }
    if let Some(owner) = sub(object, "owner", act) {
        out.insert("owner_name".into(), scalar(prop(owner, "username", act)));
    }
    Json::Object(out)
}

fn state(player: &mut Player) -> Json {
    let timeline = player.current_frame();
    player.mutate_with_update_context(|ctx| {
        let Some(clip) = ctx.stage.root_clip() else { return json!({"timeline": timeline}); };
        let Some(root) = clip.object1() else { return json!({"timeline": timeline}); };
        let mut act = Activation::from_nothing(ctx, ActivationIdentifier::root("tank-read"), clip);
        let global = fields(root, &["WIDTH", "HEIGHT", "SCALE", "TANKS", "aliveCount", "frozen", "endCount",
            "resetCount", "numberOfCrates", "crateTimer", "AIEnabled", "NUMBEROFFRAMESFROZEN"], &mut act);
        let mut tanks = Vec::new();
        let mut entities = Vec::new();
        if let Some(game) = sub(root, "game", &mut act) {
            for index in 0..2 {
                if let Some(tank) = sub(game, &format!("tank{index}"), &mut act) {
                    tanks.push(dump_scalars(tank, &mut act));
                } else { tanks.push(Json::Null); }
            }
            for container in [Some(game), sub(game, "mazebg", &mut act)].into_iter().flatten() {
                let mut names: Vec<String> = container.get_keys(&mut act, false).into_iter()
                    .map(|name| name.to_utf8_lossy().into_owned()).collect();
                names.sort();
                for name in names {
                    let lower = name.to_ascii_lowercase();
                    if !["bullet", "laser", "frag", "gatling", "deathray", "homing", "mine", "remote",
                        "electricbullet", "crate", "aimer", "shield", "eltoro"].iter().any(|p| lower.starts_with(p)) {
                        continue;
                    }
                    if let Some(child) = sub(container, &name, &mut act) {
                        if child.as_display_object().is_some() {
                            let mut item = dump_scalars(child, &mut act);
                            item["name"] = json!(name);
                            entities.push(item);
                        }
                    }
                }
            }
        }
        let width = global["WIDTH"].as_f64().unwrap_or(0.0).clamp(0.0, 64.0) as usize;
        let height = global["HEIGHT"].as_f64().unwrap_or(0.0).clamp(0.0, 64.0) as usize;
        let mut maze = Vec::new();
        if let Some(grid) = sub(root, "maze", &mut act) {
            for x in 0..width {
                let mut column = Vec::new();
                if let Some(col) = sub(grid, &x.to_string(), &mut act) {
                    for y in 0..height {
                        let mut cell = Vec::new();
                        if let Some(values) = sub(col, &y.to_string(), &mut act) {
                            for i in 0..3 { cell.push(scalar(prop(values, &i.to_string(), &mut act))); }
                        }
                        column.push(cell);
                    }
                }
                maze.push(column);
            }
        }
        json!({"timeline": timeline, "root": global, "tanks": tanks, "entities": entities, "maze": maze})
    })
}

fn call_root(player: &mut Player, method: &str, argument: &str) -> Result<(), String> {
    player.mutate_with_update_context(|ctx| {
        let clip = ctx.stage.root_clip().ok_or("root clip missing")?;
        let root = clip.object1().ok_or("AVM1 root missing")?;
        let mut act = Activation::from_nothing(ctx, ActivationIdentifier::root("tank-menu"), clip);
        let method = AvmString::new_utf8(act.gc(), method);
        let arg = Value::String(AvmString::new_utf8(act.gc(), argument));
        root.call_method(method, &[arg], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
        Player::run_actions(act.context);
        Ok(())
    })
}

fn key(code: u32) -> Result<KeyDescriptor, String> {
    let (physical_key, logical_key) = match code {
        37 => (PhysicalKey::ArrowLeft, LogicalKey::Named(NamedKey::ArrowLeft)),
        38 => (PhysicalKey::ArrowUp, LogicalKey::Named(NamedKey::ArrowUp)),
        39 => (PhysicalKey::ArrowRight, LogicalKey::Named(NamedKey::ArrowRight)),
        40 => (PhysicalKey::ArrowDown, LogicalKey::Named(NamedKey::ArrowDown)),
        77 => (PhysicalKey::KeyM, LogicalKey::Character('m')),
        81 => (PhysicalKey::KeyQ, LogicalKey::Character('q')),
        69 => (PhysicalKey::KeyE, LogicalKey::Character('e')),
        83 => (PhysicalKey::KeyS, LogicalKey::Character('s')),
        70 => (PhysicalKey::KeyF, LogicalKey::Character('f')),
        68 => (PhysicalKey::KeyD, LogicalKey::Character('d')),
        _ => return Err(format!("Unsupported game key: {code}")),
    };
    Ok(KeyDescriptor { physical_key, logical_key, key_location: KeyLocation::Standard })
}

struct Session {
    player: Arc<Mutex<Player>>,
    executor: NullExecutor,
    keys: BTreeSet<u32>,
    frame: u64,
    teacher_rng: crate::avm_rng::AvmRng,
    sensor_rng: crate::avm_rng::AvmRng,
}

impl Session {
    fn new(path: &Path, seed: u64, mode: &str, observe: bool) -> Result<Self, String> {
        let executor = NullExecutor::new();
        let movie = crate::tag_utils::movie_from_path(path, None).map_err(|e| e.to_string())?;
        let date = chrono::Utc.timestamp_micros(1_600_000_000_000_001 + (seed % 4_000_000_000) as i64)
            .single().ok_or("bad seed date")?.fixed_offset();
        let navigator = NullNavigatorBackend::with_base_path(path.parent().ok_or("movie parent missing")?, &executor)
            .map_err(|e| e.to_string())?;
        let player = PlayerBuilder::new().with_movie(movie).with_navigator(navigator)
            .with_locale(DeterministicLocaleBackend::new(date)).with_load_behavior(LoadBehavior::Blocking)
            .with_max_execution_duration(std::time::Duration::from_secs(30)).build();
        player.lock().unwrap().preload(&mut ExecutionLimit::none());
        player.lock().unwrap().set_is_playing(true);
        player.lock().unwrap().handle_event(PlayerEvent::FocusGained);
        let mut session = Self { player, executor, keys: BTreeSet::new(), frame: 0,
                                 teacher_rng: crate::avm_rng::AvmRng::default(),
                                 sensor_rng: crate::avm_rng::AvmRng::default() };
        for _ in 0..8 { session.advance(); }
        let label = if mode == "laika" { "controls1Player" } else if mode == "duel" { "controls2Player" }
            else { return Err("mode must be laika or duel".into()); };
        call_root(&mut session.player.lock().unwrap(), "gotoAndPlay", label)?;
        let control_frame = if mode == "laika" { 7 } else { 19 };
        for _ in 0..100 {
            if session.player.lock().unwrap().current_frame() == Some(control_frame) { break; }
            session.advance();
        }
        session.apply_keys(if mode == "laika" { vec![77] } else { vec![77,81] })?;
        session.advance();
        session.apply_keys(vec![])?;
        for _ in 0..200 {
            session.advance();
            let current = session.snapshot();
            if current["timeline"] == 53 && current["tanks"][0]["alive"] == true &&
                current["tanks"][1]["alive"] == true && current["tanks"][0]["forwardSpeed"].is_number() {
                session.frame = 0;
                HITS.with(|events| events.borrow_mut().clear());
                FIRES.with(|events| events.borrow_mut().clear());
                if observe { install_hit_observer(&mut session.player.lock().unwrap()); }
                return Ok(session);
            }
        }
        Err(format!("Original game did not reach playable state: {}", session.snapshot()))
    }

    fn advance(&mut self) {
        HOST_FRAME.with(|frame| frame.set(self.frame + 1));
        self.executor.run();
        {
            let mut player = self.player.lock().unwrap();
            player.run_frame();
            player.update_timers(FloatDuration::from_millis(40.0));
            player.update_sockets();
            player.update_net_connections();
        }
        self.executor.run();
        self.frame += 1;
    }

    fn apply_keys(&mut self, keys: Vec<u32>) -> Result<(), String> {
        let wanted: BTreeSet<u32> = keys.into_iter().collect();
        for code in wanted.iter() { key(*code)?; }
        let mut player = self.player.lock().unwrap();
        for code in self.keys.difference(&wanted) { player.handle_event(PlayerEvent::KeyUp { key: key(*code)? }); }
        for code in wanted.difference(&self.keys) { player.handle_event(PlayerEvent::KeyDown { key: key(*code)? }); }
        self.keys = wanted;
        Ok(())
    }

    fn snapshot(&self) -> Json {
        let mut result = state(&mut self.player.lock().unwrap());
        result["frame"] = json!(self.frame);
        result["hits"] = HITS.with(|events| json!(*events.borrow()));
        result["fires"] = FIRES.with(|events| json!(*events.borrow()));
        result
    }

    fn round_frozen(&self) -> bool {
        self.player.lock().unwrap().mutate_with_update_context(|ctx| {
            let Some(clip) = ctx.stage.root_clip() else { return false; };
            let Some(root) = clip.object1() else { return false; };
            let mut act = Activation::from_nothing(ctx, ActivationIdentifier::root("tank-settle"), clip);
            matches!(prop(root, "frozen", &mut act), Value::Bool(true))
        })
    }

    fn shot_paths(&mut self) -> Result<Json, String> {
        let sensor_rng = &mut self.sensor_rng;
        self.player.lock().unwrap().mutate_with_update_context(|ctx| {
            let clip = ctx.stage.root_clip().ok_or("root missing")?;
            let root = clip.object1().ok_or("root object missing")?;
            let mut act = Activation::from_nothing(ctx, ActivationIdentifier::root("shot-path-query"), clip);
            let game_rng = std::mem::replace(act.context.rng, std::mem::take(sensor_rng));
            let result = (|| -> Result<Json, String> {
                let game = sub(root, "game", &mut act).ok_or("game missing")?;
                let tank = sub(game, "tank0", &mut act).ok_or("tank missing")?;
                if !matches!(prop(tank, "alive", &mut act), Value::Bool(true)) {
                    return Ok(json!({"alive": false, "paths": []}));
                }
                // 独立实例只调用原版几何函数，不运行专家决策或设置按键。
                if sub(root, "__tank_rl_sensor", &mut act).is_none() {
                    let method = AvmString::new_utf8(act.gc(), "getNextHighestDepth");
                    let depth = root.call_method(method, &[], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    let method = AvmString::new_utf8(act.gc(), "attachMovie");
                    let symbol = Value::String(AvmString::new_utf8(act.gc(), "tankTroubleAI"));
                    let name = Value::String(AvmString::new_utf8(act.gc(), "__tank_rl_sensor"));
                    root.call_method(method, &[symbol, name, depth], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    Player::run_actions(act.context);
                }
                let sensor = sub(root, "__tank_rl_sensor", &mut act).ok_or("sensor clip missing")?;
                let name = AvmString::new_utf8(act.gc(), "myTank");
                sensor.define_value(act.gc(), name, tank.into(), Attribute::empty());
                let Value::Number(angle) = prop(tank, "_rotation", &mut act) else { return Err("tank angle missing".into()); };
                let mut paths = Vec::new();
                for offset in [-10.0, 0.0, 10.0] {
                    let method = AvmString::new_utf8(act.gc(), "checkBulletPath");
                    let value = sensor.call_method(method, &[Value::Number(angle + offset)], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    let prediction = value.as_object(&mut act).ok_or("shot prediction missing")?;
                    let mut entry = fields(prediction, &["result", "time", "closest"], &mut act);
                    entry["angle_offset"] = json!(offset);
                    paths.push(entry);
                }
                Ok(json!({"alive": true, "paths": paths}))
            })();
            *sensor_rng = std::mem::replace(act.context.rng, game_rng);
            result
        })
    }

    fn teacher_action(&mut self) -> Result<Json, String> {
        let teacher_rng = &mut self.teacher_rng;
        self.player.lock().unwrap().mutate_with_update_context(|ctx| {
            let clip = ctx.stage.root_clip().ok_or("root missing")?;
            let root = clip.object1().ok_or("root object missing")?;
            let mut act = Activation::from_nothing(ctx, ActivationIdentifier::root("teacher-query"), clip);
            // 专家查询使用独立 RNG；不消耗实际对局的随机序列。
            let game_rng = std::mem::replace(act.context.rng, std::mem::take(teacher_rng));
            let result = (|| -> Result<Json, String> {
                if sub(root, "__tank_rl_teacher", &mut act).is_none() {
                    let method = AvmString::new_utf8(act.gc(), "getNextHighestDepth");
                    let depth = root.call_method(method, &[], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    let method = AvmString::new_utf8(act.gc(), "attachMovie");
                    let symbol = Value::String(AvmString::new_utf8(act.gc(), "tankTroubleAI"));
                    let name = Value::String(AvmString::new_utf8(act.gc(), "__tank_rl_teacher"));
                    root.call_method(method, &[symbol, name, depth], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    Player::run_actions(act.context);
                }
                let advisor = sub(root, "__tank_rl_teacher", &mut act).ok_or("teacher clip missing")?;
                let game = sub(root, "game", &mut act).ok_or("game missing")?;
                let tank = sub(game, "tank0", &mut act).ok_or("tank missing")?;
                if !matches!(prop(tank, "alive", &mut act), Value::Bool(true)) { return Err("teacher requires alive player".into()); }
                let name = AvmString::new_utf8(act.gc(), "myTank");
                advisor.define_value(act.gc(), name, tank.into(), Attribute::empty());
                let control_names = ["forward", "backup", "turnLeft", "turnRight", "fire", "i"];
                let old: Vec<_> = control_names.iter().map(|name| prop(tank, name, &mut act)).collect();
                let answer = (|| -> Result<Json, String> {
                    let method = AvmString::new_utf8(act.gc(), "makeDecisionsAndUpdateGoal");
                    let update = advisor.call_method(method, &[], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    if matches!(update, Value::Bool(true)) {
                        let method = AvmString::new_utf8(act.gc(), "decideActionsToAchieveGoal");
                        advisor.call_method(method, &[], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    }
                    let method = AvmString::new_utf8(act.gc(), "setInputToDoActions");
                    advisor.call_method(method, &[], &mut act, ExecutionReason::Special).map_err(|e| e.to_string())?;
                    Ok(fields(tank, &["forward", "backup", "turnLeft", "turnRight", "fire"], &mut act))
                })();
                for (name, value) in control_names.iter().zip(old) {
                    let name = AvmString::new_utf8(act.gc(), *name);
                    tank.define_value(act.gc(), name, value, Attribute::empty());
                }
                answer
            })();
            *teacher_rng = std::mem::replace(act.context.rng, game_rng);
            result
        })
    }
}

/// JSONL 协议：每个请求严格对应一个响应，错误不会默默退回替代环境。
pub fn run() {
    let Some(path) = std::env::args().nth(1) else { eprintln!("Usage: tank_rl <original.swf>"); return; };
    let path = PathBuf::from(path).canonicalize().expect("SWF path must exist");
    let mut session: Option<Session> = None;
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break; };
        let response: Result<Json, String> = (|| {
            let request: Json = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            match request["op"].as_str().unwrap_or("") {
                "reset" => {
                    let next = Session::new(&path, request["seed"].as_u64().unwrap_or(0), request["mode"].as_str().unwrap_or("laika"), request["observe"].as_bool().unwrap_or(false))?;
                    let result = next.snapshot(); session = Some(next); Ok(result)
                }
                "step" => {
                    let current = session.as_mut().ok_or("reset required")?;
                    let keys: Vec<u32> = request["keys"].as_array().ok_or("keys array required")?.iter()
                        .map(|v| v.as_u64().map(|x| x as u32).ok_or("bad key".to_string())).collect::<Result<_,_>>()?;
                    current.apply_keys(keys)?;
                    let frames = request["frames"].as_u64().unwrap_or(1);
                    if !(1..=250).contains(&frames) { return Err("frames must be 1..250".into()); }
                    for _ in 0..frames { current.advance(); }
                    Ok(current.snapshot())
                }
                "state" => Ok(session.as_ref().ok_or("reset required")?.snapshot()),
                "teacher" => session.as_mut().ok_or("reset required")?.teacher_action(),
                "shot_paths" => session.as_mut().ok_or("reset required")?.shot_paths(),
                "settle" => {
                    let current = session.as_mut().ok_or("reset required")?;
                    if current.snapshot()["tanks"][0]["alive"] != false {
                        return Err("settle requires dead player 0".into());
                    }
                    current.apply_keys(vec![])?;
                    // 完整执行每个原版帧，仅省去中间帧的 JSON 状态序列化与往返通信。
                    for _ in 0..500 {
                        if current.round_frozen() { break; }
                        current.advance();
                    }
                    if !current.round_frozen() { return Err("settlement exceeded 500 frames".into()); }
                    Ok(current.snapshot())
                }
                _ => Err("Unknown operation".into()),
            }
        })();
        let response = match response { Ok(value) => json!({"ok":true,"state":value}), Err(error) => json!({"ok":false,"error":error}) };
        writeln!(stdout, "{}", response).expect("write response");
        stdout.flush().expect("flush response");
    }
}
