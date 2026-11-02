// The loading screen (splash.html, splash.css): it goes once the editor is up, or after MAX_MS whatever happens.
(function () {
    "use strict";
    var root = document.documentElement;
    var MAX_MS = 15000;
    var t0 = performance.now();

    // the editor is up: the app set up and its own splash removed or its spinner off
    function editorReady() {
        var app = window.app;
        return !document.getElementById("splash-loader") ||
            !!(app && app.extensionManager && app.extensionManager.spinner === false && app.canvas);
    }

    function dismissWhenReady() {
        var layer = document.getElementById("loom-splash");
        if (!layer) return;
        if (editorReady() || performance.now() - t0 > MAX_MS) {
            root.classList.add("loom-gone");
            setTimeout(function () {
                layer.remove();
                root.classList.remove("loom-gone");
            }, 500);
            return;
        }
        setTimeout(dismissWhenReady, 120);
    }

    document.addEventListener("DOMContentLoaded", function () { setTimeout(dismissWhenReady, 120); });
})();
