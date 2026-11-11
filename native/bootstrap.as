// 只选择原版模式；电影、键盘、画面、声音和游戏逻辑均由原版SWF执行。
stop();
var bootstrap = this;
bootstrap.frames = 0;
bootstrap.loaded = false;
bootstrap.selected = false;
bootstrap.createEmptyMovieClip("original", 0);
bootstrap.original._lockroot = true;
var listener = {};
listener.onLoadInit = function(movie) {
    movie._lockroot = true;
    bootstrap.loaded = true;
    trace("TANK_START original loaded");
};
var loader = new MovieClipLoader();
loader.addListener(listener);
loader.loadClip("TankTrouble_v4.0.swf", bootstrap.original);
bootstrap.onEnterFrame = function() {
    if (!bootstrap.loaded) { return; }
    if (bootstrap.selected) {
        var wanted = bootstrap.gameMode == "controls1Player" ? 7 : (bootstrap.gameMode == "controls2Player" ? 19 : 31);
        if (bootstrap.original._currentframe == wanted) {
            trace("TANK_START ready " + bootstrap.gameMode + " frame=" + wanted);
            bootstrap.onEnterFrame = undefined;
        }
        return;
    }
    bootstrap.frames++;
    if (bootstrap.frames < 8) { return; }
    var mode = bootstrap.gameMode;
    if (mode != "controls1Player" && mode != "controls2Player" && mode != "controls3Player") {
        mode = "controls1Player";
    }
    bootstrap.original.gotoAndPlay(mode);
    bootstrap.gameMode = mode;
    bootstrap.selected = true;
};
