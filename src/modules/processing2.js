import {
    calcResize,
    loadVideo,
    getCanvas,
    emitEvent,
    requestIdleCB,
    canvToBlob,
} from "./helpers";
import { removeBlurryStart } from "./style";
import { STATUSES } from "../constants.js";

const FRAME_RATE = 1000 / 25; // 25 fps

// threshold for number of consecutive frames that need to be positive for the image to be considered positive
const POSITIVE_THRESHOLD = 1; //at 25 fps, this is 0.04 seconds of consecutive positive detections
// threshold for number of consecutive frames that need to be negative for the image to be considered negative
const NEGATIVE_THRESHOLD = 3; //at 25 fps, this is 0.12 seconds of consecutive negative detections
// versioning approach supersedes fixed timers. Small fallback delay is still possible but not required.
const SRC_STABLE_TIMEOUT = 0;
/**
 * Object containing the possible results of image processing.
 * @typedef {Object} RESULTS
 * @property {string} CLEAR - Indicates that the image is clear and safe to display.
 * @property {string} NSFW - Indicates that the image contains NSFW content and should be blurred.
 * @property {string} FACE - Indicates that the image contains a face and should be blurred.
 * @property {string} ERROR - Indicates that an error occurred during processing.
 */
const RESULTS = {
    CLEAR: "CLEAR",
    NSFW: "NSFW",
    FACE: "FACE",
    ERROR: "ERROR",
};

let activeFrame = false;
let canv, ctx;

const DEBUG = true;
const debugLog = (label, node, extra = "") => {
    if (!DEBUG) return;
    console.log(
        `[HB] ${label} src=${node?.src?.slice(0, 80)} t=${performance.now().toFixed(0)} ${extra}`
    );
};

const processImage = (node, STATUSES, opts = {}) => {
    try {
        // no log needed here
        node.dataset.HBstatus = STATUSES.PROCESSING;
        // increment and tag version for this detection
        node.HBver = (node.HBver || 0) + 1;
        const myVer = node.HBver;

        chrome.runtime.sendMessage(
            {
                type: "imageDetection",
                version: myVer,
                image: {
                    src: node.src,
                    width: node.width || node.naturalWidth,
                    height: node.height || node.naturalHeight,
                },
            },
            (response) => {
                const { version, result, type, scores, faces } = response;

                if (version !== node.HBver) {
                    debugLog("STALE", node, `ver=${version}`);
                    return; // ignore stale results
                }

                if (type === "error") {
                    console.warn("HB==Error while processing image", response);
                    node.dataset.HBstatus = STATUSES.ERROR;
                    return;
                }

                const isUnsafe = result === "face" || result === "nsfw";

                if (isUnsafe) {
                    // Cancel any pending safe-unblur timer
                    if (node.HBunblurTimeout) {
                        clearTimeout(node.HBunblurTimeout);
                        node.HBunblurTimeout = null;
                    }
                    debugLog("UNSAFE", node);
                    debugLog("Marked UNSAFE, adding hb-blur", node.src);
                    node.classList.add("hb-blur");
                    node.dataset.HBresult = result;
                    removeBlurryStart(node);
                } else {
                    // safe result for current src
                    if (
                        !node.classList.contains("hb-blur") &&
                        node.dataset.HBresult !== "nsfw" &&
                        node.dataset.HBresult !== "face"
                    ) {
                        node.classList.remove("hb-blur");
                        delete node.dataset.HBresult;
                        removeBlurryStart(node);
                        debugLog("UNBLUR", node);
                    }
                }

                if (scores) {
                    // store top class scores as JSON string (trimmed) for developer inspection
                    node.setAttribute(
                        "data--h-bscores",
                        JSON.stringify(scores.slice(0, 3))
                    );
                }

                if (faces && faces.length) {
                    // store simplified face data for debugging
                    const faceData = faces.slice(0, 3).map((f) => {
                        return {
                            confidence: Number(
                                (f?.score ?? f?.confidence ?? 0).toFixed(2)
                            ),
                            gender: f?.gender,
                            gScore: Number(
                                (f?.genderScore ?? f?.genderScore ?? 0).toFixed(
                                    2
                                )
                            ),
                            age: f?.age ? Math.round(f.age) : undefined,
                        };
                    });
                    node.setAttribute(
                        "data--h-bfaces",
                        JSON.stringify(faceData)
                    );
                }
            }
        );
    } catch (e) {
        console.log("HB==ERROR", e);
    }
};

const processFrame = async (video, { width, height }) => {
    if (!video || video.ended) {
        return;
    }
    return new Promise(async (resolve, reject) => {
        try {
            ctx.drawImage(video, 0, 0, width, height);

            const blob = await canvToBlob(canv, {
                type: "image/jpeg",
                quality: 0.6,
            });
            let data = URL.createObjectURL(blob);
            chrome.runtime.sendMessage(
                {
                    type: "videoDetection",
                    frame: {
                        data: data,
                        timestamp: video.currentTime,
                    },
                },
                (response) => {
                    // revoke the object url to free up memory
                    URL.revokeObjectURL(data);
                    resolve(response);
                }
            );
        } catch (e) {
            reject(e);
        }
    });
};

const videoDetectionLoop = async (video, { width, height }) => {
    // get the current timestamp
    const currTime = performance.now();

    if (!video?.HBprevTime) {
        video.HBprevTime = currTime;
    }

    // calculate the time difference
    const diffTime = currTime - video.HBprevTime;

    if (video.dataset.HBstatus === STATUSES.DISABLED) {
        video.classList.remove("hb-blur");
    }
    if (
        !video.ended &&
        !video.paused &&
        video.dataset.HBstatus !== STATUSES.DISABLED
    ) {
        try {
            if (diffTime >= FRAME_RATE) {
                // store the current timestamp
                video.HBprevTime = currTime;

                if (!activeFrame) {
                    activeFrame = true;
                    processFrame(video, { width, height })
                        .then(({ result, timestamp }) => {
                            if (result === "error") {
                                throw new Error("HB==Error from processFrame");
                            }

                            // if frame was skipped, don't process it
                            if (result === "skipped") {
                                // console.log( "skipped frame");
                                return;
                            }

                            // if the frame is too old, don't process it
                            if (video.currentTime - timestamp > 0.5) {
                                // console.log("too old frame");
                                return;
                            }

                            // process the result
                            processVideoDetections(result, video);
                        })
                        .catch((error) => {
                            throw error;
                        })
                        .finally(() => {
                            activeFrame = false;
                        });
                }
            }
        } catch (error) {
            console.log("HB==Video detection loop error", error, video);
            video.dataset.HBerrored =
                parseInt(video.dataset.HBerrored ?? 0) + 1;
        }
    }

    if (video.dataset.HBerrored > 10) {
        // remove onplay listener
        video.onplay = null;
        cancelAnimationFrame(video.HBrafId);
        video.removeAttribute("crossorigin");
        return;
    }
    if (!video.paused) {
        video.HBrafId = requestAnimationFrame(() =>
            videoDetectionLoop(video, { width, height })
        );
    } else {
        video.onplay = () => {
            video.HBrafId = requestAnimationFrame(() =>
                videoDetectionLoop(video, { width, height })
            );
        };
    }
};
const processVideo = async (node) => {
    try {
        node.dataset.HBstatus = STATUSES.LOADING;
        await loadVideo(node);
        node.dataset.HBstatus = STATUSES.PROCESSING;
        const { newWidth, newHeight } = calcResize(
            node.videoWidth ?? node.clientWidth,
            node.videoHeight ?? node.clientHeight,
            "video"
        );
        if (!canv) {
            canv = getCanvas(newWidth, newHeight, true);
            ctx = canv.getContext("2d", {
                alpha: false,
                willReadFrequently: true,
            });
        }
        // set the width and height of the video
        node.width = newWidth;
        node.height = newHeight;

        if (canv.width !== newWidth || canv.height !== newHeight) {
            canv.width = newWidth;
            canv.height = newHeight;
        }

        // Keep the temporary blur until we have enough evidence to unblur/blur permanently.

        // start the video detection loop but don't block the main thread
        requestIdleCB(() => {
            videoDetectionLoop(node, { width: newWidth, height: newHeight });
        });
    } catch (e) {
        console.log("HB== processVideo error", e);
    }
};

const processVideoDetections = (result, video) => {
    const prevResult = video.dataset.HBresult;
    const isPrevResultClear = prevResult === RESULTS.CLEAR || !prevResult;
    const currentPositiveCount = parseInt(video.HBpositiveCount ?? 0);
    const currentNegativeCount = parseInt(video.HBnegativeCount ?? 0);
    let shouldBlur = null;

    if (result === "nsfw") {
        video.dataset.HBresult = RESULTS.NSFW;
        video.HBpositiveCount = currentPositiveCount + !isPrevResultClear;
        video.HBnegativeCount = 0;
        // if the positive count is greater than the threshold (i.e it's not a momentary blip), add the blur
        if (currentPositiveCount + !isPrevResultClear >= POSITIVE_THRESHOLD) {
            // video.pause()
            shouldBlur = true;
            video.HBpositiveCount = 0;
        }
    } else if (result === "face") {
        video.dataset.HBresult = RESULTS.FACE;
        video.HBpositiveCount = currentPositiveCount + !isPrevResultClear;
        video.HBnegativeCount = 0;
        // if the positive count is greater than the threshold (i.e it's not a momentary blip), add the blur
        if (currentPositiveCount + !isPrevResultClear >= POSITIVE_THRESHOLD) {
            // video.pause()
            shouldBlur = true;
            video.HBpositiveCount = 0;
        }
    } else {
        video.dataset.HBresult = RESULTS.CLEAR;
        video.HBnegativeCount = currentNegativeCount + isPrevResultClear;
        video.HBpositiveCount = 0;
        // if the negative count is greater than the threshold (i.e it's not a momentary blip), remove the blur
        if (currentNegativeCount + isPrevResultClear >= NEGATIVE_THRESHOLD) {
            shouldBlur = false;
            video.HBnegativeCount = 0;
        }
    }

    if (shouldBlur !== null) {
        if (shouldBlur) {
            // Add permanent blur first
            video.classList.add("hb-blur");
            // Then remove the temporary blur if it's still there
            if (video.classList.contains("hb-blur-temp"))
                removeBlurryStart(video);
        } else {
            // Safe content – remove temp blur first
            if (video.classList.contains("hb-blur-temp"))
                removeBlurryStart(video);
            video.classList.remove("hb-blur");
        }
    }
};
export { processImage, processVideo };
