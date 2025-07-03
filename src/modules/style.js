// style.js
// This module exports the style sheet and blur effect functions

import { emitEvent, listenToEvent } from "./helpers.js";

const BLURRY_START_MODE_TIMEOUT = 7000; // TODO: make this a setting maybe?
let hbStyleSheet, blurryStartStyleSheet, _settings;

const initStylesheets = ({ detail }) => {
    _settings = detail;
    // console.log("HB==INIT STYLESHEETS")
    hbStyleSheet = document.createElement("style");
    hbStyleSheet.id = "hb-stylesheet";
    document.head.appendChild(hbStyleSheet);
};

const setStyle = ({ detail: settings }) => {
    _settings = settings;
    // console.log("HB==SET STYLE")
    if (!hbStyleSheet) {
        initStylesheets();
    }
    if (!_settings.shouldDetect()) {
        hbStyleSheet.innerHTML = "";
        return;
    }
    const shouldBlurImages = _settings.shouldBlurImages();
    const shouldBlurVideos = _settings.shouldBlurVideos();
    const shouldUnblurImagesOnHover = _settings.shouldUnblurImages();
    const shouldUnblurVideosOnHover = _settings.shouldUnblurVideos();

    let blurSelectors = [];
    if (shouldBlurImages) blurSelectors.push("img" + ".hb-blur");
    if (shouldBlurVideos) blurSelectors.push("video" + ".hb-blur");
    blurSelectors = blurSelectors.join(", ");

    let unblurSelectors = [];
    if (shouldUnblurImagesOnHover)
        unblurSelectors.push("img" + ".hb-blur:hover");
    if (shouldUnblurVideosOnHover)
        unblurSelectors.push("video" + ".hb-blur:hover");
    unblurSelectors = unblurSelectors.join(", ");
    hbStyleSheet.innerHTML = `
    ${blurSelectors} {
      filter: blur(${_settings.getBlurAmount()}px) ${
          _settings.isGray() ? "grayscale(100%)" : ""
      } !important;
      transition: filter 0.1s ease !important;
      opacity: unset !important;
    }
	
  `;
    if (unblurSelectors) {
        hbStyleSheet.innerHTML += `
		${unblurSelectors} {
			filter: blur(0px) ${_settings.isGray() ? "grayscale(0%)" : ""} !important;
			transition: filter 0.5s ease !important;
			transition-delay: 1s !important;
		  }
	`;
    }

    hbStyleSheet.innerHTML += `
	.hb-blur-temp { 
		filter: blur(${_settings.getBlurAmount()}px) ${
            _settings.isGray() ? "grayscale(100%)" : ""
        } !important;
		transition: filter 0.1s ease !important;
		opacity: unset !important;
	}

	#hb-in-canvas {
		display: none !important;
		visibility: hidden !important;
	}

    /* Visual indicators for blur reasons */
    img.hb-blur[data--h-bresult="nsfw"],
    video.hb-blur[data--h-bresult="nsfw"] {
        filter: blur(${_settings.getBlurAmount()}px) ${
            _settings.isGray() ? "grayscale(100%)" : ""
        } drop-shadow(0 0 0 rgba(255,0,0,0.95)) !important;
    }

    img.hb-blur[data--h-bresult="face"],
    video.hb-blur[data--h-bresult="face"] {
        filter: blur(${_settings.getBlurAmount()}px) ${
            _settings.isGray() ? "grayscale(100%)" : ""
        } drop-shadow(0 0 0 rgba(255,165,0,0.95)) !important;
    }
  `;
};
const applyBlurryStart = (node) => {
    // Always apply blur on start for safety - images should be blurred until confirmed safe
    if (_settings?.shouldDetect()) {
        node.classList.add("hb-blur-temp");
        // concise debug log
        console.log(
            `[HB] TEMP src=${node?.src?.slice(0, 80)} t=${performance.now().toFixed(0)}`
        );
    }
};

const removeBlurryStart = (node) => {
    node.classList.remove("hb-blur-temp");
};

const attachStyleListener = () => {
    listenToEvent("settingsLoaded", initStylesheets);
    listenToEvent("toggleOnOffStatus", setStyle);
    listenToEvent("changeBlurAmount", setStyle);
    listenToEvent("changeGray", setStyle);
};

export { attachStyleListener, applyBlurryStart, removeBlurryStart };
