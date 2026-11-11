//! 原版SWF的原生渲染窗口。输入与帧推进仍复用训练宿主，不从状态重绘图形。
mod native_window {
    use super::*;
    use ruffle_render::backend::ViewportDimensions;
    use ruffle_render_wgpu::backend::WgpuRenderBackend;
    use ruffle_render_wgpu::wgpu;
    use winit::application::ApplicationHandler;
    use winit::dpi::PhysicalSize;
    use winit::event::{ElementState, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey as WindowKey, Key as WindowLogicalKey, NamedKey as WindowNamedKey};
    use winit::window::{Window, WindowId};

    struct App {
        session: Option<Session>,
        window: Option<Arc<Window>>,
        movie: PathBuf,
        human_keys: BTreeSet<u32>,
        human_pulses: BTreeSet<u32>,
        human_codes: BTreeSet<u32>,
        round: u64,
        round_start: u64,
        awaiting_round: bool,
    }

    impl App {
        fn state(&mut self) -> Result<Json, String> {
            let session = self.session.as_ref().ok_or("reset required")?;
            let mut state = session.snapshot();
            if state["root"]["frozen"] == true && state["root"]["aliveCount"].as_f64().unwrap_or(2.0) <= 1.0 {
                self.awaiting_round = true;
            }
            if self.awaiting_round && state["root"]["frozen"] == false && state["root"]["aliveCount"].as_f64() == Some(2.0) {
                self.round += 1;
                self.round_start = session.frame;
                self.awaiting_round = false;
            }
            state["round"] = json!(self.round);
            state["round_frame"] = json!(session.frame - self.round_start);
            state["human_keys"] = json!(self.human_keys);
            state["applied_keys"] = json!(session.keys);
            Ok(state)
        }

        fn request(&mut self, request: Json) -> Result<Json, String> {
            match request["op"].as_str().unwrap_or("") {
                "reset" => {
                    self.session = None;
                    let window = self.window.as_ref().ok_or("window not initialized")?;
                    let size = window.inner_size();
                    // 窗口的生存期长于Session内GPU表面；关闭时先释放Session。
                    let target = unsafe { wgpu::SurfaceTargetUnsafe::from_window(window.as_ref()) }.map_err(|e| e.to_string())?;
                    let renderer = unsafe {
                        WgpuRenderBackend::for_window_unsafe(target, (size.width.max(1), size.height.max(1)),
                            wgpu::Backends::PRIMARY, wgpu::PowerPreference::LowPower, Some(Box::new(window.clone())))
                    }.map_err(|e| e.to_string())?;
                    let mode = request["mode"].as_str().unwrap_or("laika");
                    let session = Session::new(&self.movie, request["seed"].as_u64().unwrap_or(0), mode,
                        request["observe"].as_bool().unwrap_or(true), Some(renderer))?;
                    session.player.lock().unwrap().set_viewport_dimensions(ViewportDimensions {
                        width: size.width.max(1), height: size.height.max(1), scale_factor: window.scale_factor() });
                    self.human_keys.clear();
                    self.human_pulses.clear();
                    self.human_codes.clear();
                    if mode == "duel" {
                        let state = session.snapshot();
                        for name in ["KEYFORWARD", "KEYBACKUP", "KEYTURNLEFT", "KEYTURNRIGHT", "KEYFIRE"] {
                            if let Some(code) = state["tanks"][1][name].as_f64() { self.human_codes.insert(code as u32); }
                        }
                    }
                    self.session = Some(session);
                    self.round = 1; self.round_start = 0; self.awaiting_round = false;
                    self.state()
                }
                "step" => {
                    let mut keys: BTreeSet<u32> = request["keys"].as_array().ok_or("keys array required")?.iter()
                        .map(|v| v.as_u64().map(|x| x as u32).ok_or("bad key".to_string())).collect::<Result<_,_>>()?;
                    // 只接收真人一侧的原版按键，不覆盖模型一侧。
                    keys.extend(self.human_keys.intersection(&self.human_codes).copied());
                    // 将两次物理帧之间的真人短按至少传入一次，松键不会造成粘连。
                    keys.extend(self.human_pulses.intersection(&self.human_codes).copied());
                    self.human_pulses.clear();
                    let session = self.session.as_mut().ok_or("reset required")?;
                    session.apply_keys(keys.into_iter().collect())?;
                    let frames = request["frames"].as_u64().unwrap_or(1);
                    if !(1..=250).contains(&frames) { return Err("invalid frame count".into()); }
                    for _ in 0..frames { session.advance(); }
                    self.state()
                }
                "state" => self.state(),
                "settle" => {
                    if !self.human_codes.is_empty() { return Err("human mode requires frame-by-frame settlement".into()); }
                    let session = self.session.as_mut().ok_or("reset required")?;
                    if session.snapshot()["tanks"][0]["alive"] != false { return Err("player0 must be dead".into()); }
                    session.apply_keys(vec![])?;
                    for _ in 0..500 { if session.round_frozen() { break; } session.advance(); }
                    if !session.round_frozen() { return Err("settlement exceeded500 frames".into()); }
                    self.state()
                }
                _ => Err("unknown operation".into()),
            }
        }
    }

    impl ApplicationHandler<String> for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_none() {
                let attributes = Window::default_attributes().with_title("Tank Trouble v4.0 - Original Flash / Model Bridge")
                    .with_inner_size(PhysicalSize::new(1000, 750));
                self.window = Some(Arc::new(event_loop.create_window(attributes).expect("create original SWF window")));
            }
        }

        fn user_event(&mut self, event_loop: &ActiveEventLoop, text: String) {
            if text == "__EOF__" { self.session = None; event_loop.exit(); return; }
            let result = serde_json::from_str::<Json>(&text).map_err(|e| e.to_string()).and_then(|request| self.request(request));
            let response = match result { Ok(state) => json!({"ok": true, "state":state}), Err(error) => json!({"ok":false,"error":error}) };
            if let Some(window) = &self.window { window.request_redraw(); }
            let mut output = io::stdout().lock();
            if writeln!(output, "{}", response).and_then(|_| output.flush()).is_err() {
                self.session = None; event_loop.exit();
            }
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::CloseRequested => { self.session = None; event_loop.exit(); }
                WindowEvent::RedrawRequested => {
                    if let Some(session) = &self.session { session.player.lock().unwrap().render(); }
                }
                WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                    if let Some(session) = &self.session {
                        session.player.lock().unwrap().set_viewport_dimensions(ViewportDimensions {
                            width:size.width, height:size.height, scale_factor:self.window.as_ref().unwrap().scale_factor() });
                    }
                    if let Some(window) = &self.window { window.request_redraw(); }
                }
                WindowEvent::Focused(false) => { self.human_keys.clear(); self.human_pulses.clear(); }
                WindowEvent::KeyboardInput { event, .. } => {
                    let physical_code = match event.physical_key {
                        WindowKey::Code(KeyCode::ArrowLeft) => Some(37), WindowKey::Code(KeyCode::ArrowUp) => Some(38),
                        WindowKey::Code(KeyCode::ArrowRight) => Some(39), WindowKey::Code(KeyCode::ArrowDown) => Some(40),
                        WindowKey::Code(KeyCode::KeyM) => Some(77), WindowKey::Code(KeyCode::KeyQ) => Some(81),
                        WindowKey::Code(KeyCode::KeyE) => Some(69), WindowKey::Code(KeyCode::KeyS) => Some(83),
                        WindowKey::Code(KeyCode::KeyF) => Some(70), WindowKey::Code(KeyCode::KeyD) => Some(68), _ => None };
                    // 原版Flash按虚拟键码读取输入；兼容方向键的逻辑事件与键盘布局。
                    let code = match &event.logical_key {
                        WindowLogicalKey::Named(WindowNamedKey::ArrowLeft) => Some(37),
                        WindowLogicalKey::Named(WindowNamedKey::ArrowUp) => Some(38),
                        WindowLogicalKey::Named(WindowNamedKey::ArrowRight) => Some(39),
                        WindowLogicalKey::Named(WindowNamedKey::ArrowDown) => Some(40),
                        WindowLogicalKey::Character(text) => match text.to_lowercase().as_str() {
                            "m" => Some(77), "q" => Some(81), "e" => Some(69), "s" => Some(83),
                            "f" => Some(70), "d" => Some(68), _ => physical_code },
                        _ => physical_code,
                    };
                    if let Some(code) = code {
                        if event.state == ElementState::Pressed {
                            self.human_keys.insert(code); self.human_pulses.insert(code);
                        } else { self.human_keys.remove(&code); }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn run() {
        let movie = PathBuf::from(std::env::args().nth(1).expect("original SWF path required"))
            .canonicalize().expect("original SWF must exist");
        let event_loop = EventLoop::<String>::with_user_event().build().expect("create native event loop");
        let proxy = event_loop.create_proxy();
        std::thread::spawn(move || {
            for text in io::stdin().lock().lines().map_while(Result::ok) {
                if proxy.send_event(text).is_err() { return; }
            }
            let _ = proxy.send_event("__EOF__".into());
        });
        let mut app = App { session:None, window:None, movie, human_keys:BTreeSet::new(), human_pulses:BTreeSet::new(), human_codes:BTreeSet::new(),
            round:1, round_start:0, awaiting_round:false };
        event_loop.run_app(&mut app).expect("run original SWF window");
    }
}

pub fn run_windowed() { native_window::run(); }
