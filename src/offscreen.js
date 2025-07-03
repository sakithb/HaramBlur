import {
    containsNsfw,
    containsGenderFace,
    Detector,
} from "./modules/detector.js";
import Queue from "./modules/queues.js";
import Settings from "./modules/settings.js";

var settings;
var queue;
var detector = new Detector();

const loadModels = async () => {
    try {
        await detector.initHuman();
        await detector.initNsfwModel();
        detector.human.events?.addEventListener("error", (e) => {
            chrome.runtime.sendMessage({ type: "reloadExtension" });
        });
    } catch (e) {
        console.log("Error loading models", e);
    }
};

const handleImageDetection = (request, sender, sendResponse) => {
    const { version } = request;
    queue.add(
        request.image,
        (resultObj) => {
            const { decision, scores } = resultObj;
            sendResponse({ version, result: decision, scores });
        },
        (error) => {
            sendResponse({ version, type: "error", message: error });
        }
    );
};
let activeFrame = false;
let frameImage = new Image();

const handleVideoDetection = async (request, sender, sendResponse) => {
    const { frame } = request;
    const { data, timestamp } = frame;
    if (activeFrame) {
        sendResponse({ result: "skipped" });
        return;
    }
    activeFrame = true;
    frameImage.onload = () => {
        runDetection(frameImage, true)
            .then((result) => {
                activeFrame = false;
                sendResponse({ type: "detectionResult", result, timestamp });
            })
            .catch((e) => {
                console.log("HB== error in detectImage", e);
                activeFrame = false;
                sendResponse({ result: "error" });
            });
    };
    frameImage.onerror = (e) => {
        console.log("HB== image error", e);
        activeFrame = false;
        sendResponse({ result: "error" });
    };
    frameImage.src = data;
};

const startListening = () => {
    settings.listenForChanges();
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.type === "imageDetection") {
            handleImageDetection(request, sender, sendResponse);
        }
        if (request.type === "videoDetection") {
            handleVideoDetection(request, sender, sendResponse);
        }
        return true;
    });
};

const runDetection = async (img, isVideo = false) => {
    if (!settings?.shouldDetect() || !img) return false;
    const tensor = detector.human.tf.browser.fromPixels(img);
    // console.log("tensors count", human.tf.memory().numTensors);
    const nsfwDetections = await detector.nsfwModelClassify(tensor);
    const strictness = settings.getStrictness() * (isVideo ? 0.75 : 1);
    activeFrame = false;

    const nsfwFlag = containsNsfw(nsfwDetections, strictness);

    if (nsfwFlag) {
        detector.human.tf.dispose(tensor);
        return { decision: "nsfw", scores: nsfwDetections };
    }
    if (!settings.shouldDetectGender()) {
        detector.human.tf.dispose(tensor);
        return { decision: false, scores: nsfwDetections };
    }
    const predictions = await detector.humanModelClassify(tensor);
    // console.log("offscreen human result", predictions);
    detector.human.tf.dispose(tensor);
    const genderFlag = containsGenderFace(
        predictions,
        settings.shouldDetectMale(),
        settings.shouldDetectFemale()
    );

    const decision = genderFlag ? "face" : false;
    return { decision, scores: nsfwDetections };
};

const init = async () => {
    let _settings = await new Promise((resolve) => {
        chrome.runtime.sendMessage({ type: "getSettings" }, (settings) => {
            resolve(settings);
        });
    });
    settings = await Settings.init(_settings["hb-settings"]);
    console.log("Settings loaded", settings);
    try {
        await loadModels();
        console.log("Models loaded", detector.human, detector.nsfwModel);
    } catch (error) {
        console.log("Error loading models", error);
        chrome.runtime.sendMessage({ type: "reloadExtension" });
        return;
    }

    queue = new Queue(runDetection);
    startListening();
};

init();
