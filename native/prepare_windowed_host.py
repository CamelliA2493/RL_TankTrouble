"""在独立Ruffle源码副本中加入原生窗口；保留训练宿主源码与二进制。"""
from pathlib import Path
import shutil
import os


def main():
    runtime = Path(os.environ.get("TANK_RUNTIME","/root/autodl-tmp/tank-runtime"))
    original = runtime / "native-sources/ruffle"
    target = runtime / "visual-sources/ruffle"
    project = Path(__file__).resolve().parents[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copytree(original, target)
    manifest = target / "Cargo.toml"
    text = manifest.read_text()
    text = text.replace('"render", "render/naga-agal"', '"render", "render/wgpu", "render/naga-pixelbender", "render/naga-agal"')
    manifest.write_text(text)
    core = target / "core/Cargo.toml"
    text = core.read_text()
    if "ruffle_render_wgpu =" not in text:
        text = text.replace("[dependencies]\n", '[dependencies]\nruffle_render_wgpu = { path = "../render/wgpu" }\nwinit = { version = "0.30", default-features = false, features = ["rwh_06"] }\n')
        text += '\n[[example]]\nname = "tank_window"\npath = "examples/tank_window.rs"\n'
        core.write_text(text)
    source = (project/"native/tank_rl.rs").read_text()
    old = 'fn new(path: &Path, seed: u64, mode: &str, observe: bool) -> Result<Self, String>'
    source = source.replace(old, 'fn new(path: &Path, seed: u64, mode: &str, observe: bool, renderer: Option<ruffle_render_wgpu::backend::WgpuRenderBackend<ruffle_render_wgpu::target::SwapChainTarget>>) -> Result<Self, String>')
    source = source.replace('let player = PlayerBuilder::new().with_movie(movie).with_navigator(navigator)', 'let mut builder = PlayerBuilder::new().with_movie(movie).with_navigator(navigator)')
    source = source.replace('.with_max_execution_duration(std::time::Duration::from_secs(30)).build();', '.with_max_execution_duration(std::time::Duration::from_secs(30));\n        if let Some(renderer) = renderer { builder = builder.with_renderer(renderer); }\n        let player = builder.build();')
    source = source.replace('request["observe"].as_bool().unwrap_or(false))?', 'request["observe"].as_bool().unwrap_or(false), None)?')
    source += '\ninclude!("tank_window.rs");\n'
    (target/"core/src/tank_rl.rs").write_text(source)
    window = (project/"native/tank_window.rs").read_text()
    # include!处不允许模块内部文档注释放在外层条目之后。
    window = window.replace("//!", "//", 1)
    (target/"core/src/tank_window.rs").write_text(window)
    examples = target/"core/examples"
    examples.mkdir(exist_ok=True)
    (examples/"tank_window.rs").write_text('#![cfg_attr(target_os="windows", windows_subsystem="windows")]\nfn main() { ruffle_core::tank_rl::run_windowed(); }\n')
    (examples/"tank_rl.rs").write_text('fn main() { ruffle_core::tank_rl::run(); }\n')
    library=target/"core/src/lib.rs"
    library_text=library.read_text()
    if "pub mod tank_rl;" not in library_text:
        library.write_text(library_text+"\npub mod tank_rl;\n")
    core_text=core.read_text()
    if 'name = "tank_rl"' not in core_text:
        core.write_text(core_text+'\n[[example]]\nname = "tank_rl"\npath = "examples/tank_rl.rs"\n')
    if (project/"native/Cargo.lock").exists():
        shutil.copy2(project/"native/Cargo.lock",target/"Cargo.lock")
    print(target)


if __name__ == "__main__":
    main()
