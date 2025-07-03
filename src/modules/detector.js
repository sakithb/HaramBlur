// detector.js
// This module exports detector functions and variables
const nsfwUrl = chrome.runtime.getURL("src/assets/models/nsfwjs/model.json");

const HUMAN_CONFIG = {
    modelBasePath: "https://cdn.jsdelivr.net/npm/@vladmandic/human/models/",
    backend: "humangl",
    // debug: true,
    cacheSensitivity: 0.9,
    warmup: "none",
    async: true,
    filter: {
        enabled: false,
        // width: 224,
        // height: 224,
    },
    face: {
        enabled: true,
        iris: { enabled: false },
        mesh: { enabled: false },
        emotion: { enabled: false },
        detector: {
            modelPath: "blazeface.json",
            maxDetected: 4,
            minConfidence: 0.15,
        },
        description: {
            enabled: true,
            modelPath: "faceres.json",
        },
    },
    body: {
        enabled: false,
    },
    hand: {
        enabled: false,
    },
    gesture: {
        enabled: false,
    },
    object: {
        enabled: false,
    },
};

const NSFW_CONFIG = {
    size: 224,
    tfScalar: 255,
    topK: 3,
    skipTime: 4000,
    skipFrames: 99,
    cacheSensitivity: 0.9,
};

const getNsfwClasses = (factor = 0) => {
    // factor is a number between 0 and 1
    // it's used to increase the threshold for nsfw classes
    // the numbers are based on trial and error
    return {
        0: {
            className: "Drawing",
            nsfw: false,
            thresh: 0.5,
        },
        1: {
            className: "Hentai",
            nsfw: true,
            // stricter: minimum 0.6 (when strictness=1) up to 1.0
            thresh: 0.6 + (1 - factor) * 0.4,
        },
        2: {
            className: "Neutral",
            nsfw: false,
            thresh: 0.5 + factor * 0.5, // increase the factor to make it less strict
        },
        3: {
            className: "Porn",
            nsfw: true,
            // stricter: 0.25–0.5 based on strictness
            thresh: 0.25 + (1 - factor) * 0.25,
        },
        4: {
            className: "Sexy",
            nsfw: true,
            // stricter: 0.5–0.65 based on strictness
            thresh: 0.5 + (1 - factor) * 0.15,
        },
    };
};

class Detector {
    constructor() {
        this._human = null;
        this._nsfwModel = null;
        this.nsfwCache = {
            predictions: [],
            timestamp: 0,
            skippedFrames: 0,
            lastInputTensor: null,
        };
    }

    get human() {
        return this._human;
    }

    get nsfwModel() {
        return this._nsfwModel;
    }

    initHuman = async () => {
        this._human = new Human.Human(HUMAN_CONFIG);
        await this._human.load();
        this._human.tf.enableProdMode();
        // warmup the model
        const tensor = this._human.tf.zeros([1, 224, 224, 3]);
        await this._human.detect(tensor);
        this._human.tf.dispose(tensor);
        console.log("HB==Human model warmed up");
    };

    humanModelClassify = async (tensor, needToResize) => {
        if (!this._human) await this.initHuman();
        return new Promise((resolve, reject) => {
            const promise = needToResize
                ? this._human.detect(tensor, {
                      filter: {
                          enabled: true,
                          width: needToResize?.newWidth,
                          height: needToResize?.newHeight,
                      },
                  })
                : this._human.detect(tensor);
            promise
                .then((res) => {
                    resolve(res);
                })
                .catch((err) => {
                    reject(err);
                });
        });
    };

    initNsfwModel = async () => {
        // load the model from indexedDB if it exists, otherwise load from url
        const indexedDBModel =
            typeof indexedDB !== "undefined" &&
            (await this._human.tf.io.listModels());

        // if the model exists in indexedDB, load it from there
        if (indexedDBModel?.["indexeddb://nsfw-model"]) {
            this._nsfwModel = await this._human.tf.loadGraphModel(
                "indexeddb://nsfw-model"
            );
        }
        // otherwise load it from the url
        else {
            this._nsfwModel = await this._human.tf.loadGraphModel(nsfwUrl);
            // save the model to indexedDB
            await this._nsfwModel.save("indexeddb://nsfw-model");
        }
        // console.log("HB==NSFW MODEL", nsfwModel);
        const tensor = this._human.tf.zeros([1, 224, 224, 3]);
        await this._nsfwModel.predict(tensor);
        this._human.tf.dispose(tensor);
        console.log("HB==NSFW model warmed up");
    };

    nsfwModelSkip = async (input, config) => {
        const tf = this._human.tf;
        let skipFrame = false;
        if (
            config.cacheSensitivity === 0 ||
            !input?.shape ||
            input?.shape.length !== 4 ||
            input?.shape[1] > 3840 ||
            input?.shape[2] > 2160
        )
            return skipFrame; // cache disabled or input is invalid or too large for cache analysis

        if (!this.nsfwCache.lastInputTensor) {
            this.nsfwCache.lastInputTensor = tf.clone(input);
        } else if (
            this.nsfwCache.lastInputTensor.shape[1] !== input.shape[1] ||
            this.nsfwCache.lastInputTensor.shape[2] !== input.shape[2]
        ) {
            // input resolution changed
            tf.dispose(this.nsfwCache.lastInputTensor);
            this.nsfwCache.lastInputTensor = tf.clone(input);
        } else {
            const t = {};
            t.diff = tf.sub(input, this.nsfwCache.lastInputTensor);
            t.squared = tf.mul(t.diff, t.diff);
            t.sum = tf.sum(t.squared);
            const diffSum = await t.sum.data();
            const diffRelative =
                diffSum[0] /
                (input.shape[1] || 1) /
                (input.shape[2] || 1) /
                255 /
                3; // squared difference relative to input resolution and averaged per channel
            tf.dispose([
                this.nsfwCache.lastInputTensor,
                t.diff,
                t.squared,
                t.sum,
            ]);
            this.nsfwCache.lastInputTensor = tf.clone(input);
            skipFrame = diffRelative <= (config.cacheSensitivity || 0);
        }
        return skipFrame;
    };

    nsfwModelClassify = async (tensor, config = NSFW_CONFIG) => {
        if (!this._human) await this.initHuman();
        if (!this._nsfwModel) await this.initNsfwModel();
        const tf = this._human.tf;
        if (!tensor) return [];
        let resized, expanded;
        try {
            const skipAllowed = await this.nsfwModelSkip(tensor, config);
            const skipFrame =
                this.nsfwCache.skippedFrames < (config.skipFrames || 0);
            const skipTime =
                (config.skipTime || 0) >
                (performance?.now?.() || Date.now()) - this.nsfwCache.timestamp;

            // if skip is not allowed or skip time is not reached or skip frame is not reached or cache is empty then run the model
            if (
                !skipAllowed ||
                !skipTime ||
                !skipFrame ||
                this.nsfwCache.predictions.length === 0
            ) {
                // if size is not 224, resize the image
                if (
                    tensor.shape[1] !== config.size ||
                    tensor.shape[2] !== config.size
                ) {
                    resized = tf.image.resizeNearestNeighbor(tensor, [
                        config.size,
                        config.size,
                    ]);
                }
                // if 3d tensor, add a dimension
                if (
                    (resized && resized.shape.length === 3) ||
                    tensor.shape.length === 3
                ) {
                    expanded = tf.expandDims(resized || tensor, 0);
                }
                const scalar = tf.scalar(config.tfScalar);
                const normalized = tf.div(
                    expanded || resized || tensor,
                    scalar
                );
                const logits = await this._nsfwModel.predict(normalized);

                this.nsfwCache.predictions = await this.getTopKClasses(
                    logits,
                    config.topK
                );
                this.nsfwCache.timestamp = performance?.now?.() || Date.now();
                this.nsfwCache.skippedFrames = 0;

                tf.dispose(
                    [scalar, normalized, logits]
                        .concat(expanded ? [expanded] : [])
                        .concat(resized ? [resized] : [])
                );
            } else {
                this.nsfwCache.skippedFrames++;
            }

            return this.nsfwCache.predictions;
        } catch (error) {
            console.error("HB==NSFW Detection Error", resized || tensor, error);
        }
    };

    getTopKClasses = async (logits, topK) => {
        const values = await logits.data();

        const valuesAndIndices = [];
        for (let i = 0; i < values.length; i++) {
            valuesAndIndices.push({ value: values[i], index: i });
        }
        valuesAndIndices.sort((a, b) => {
            return b.value - a.value;
        });
        const topkValues = new Float32Array(topK);
        const topkIndices = new Int32Array(topK);
        for (let i = 0; i < topK; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }

        const topClassesAndProbs = [];
        for (let i = 0; i < topkIndices.length; i++) {
            topClassesAndProbs.push({
                className: getNsfwClasses()?.[topkIndices[i]].className,
                probability: topkValues[i],
                id: topkIndices[i],
            });
        }
        return topClassesAndProbs;
    };
}

const containsNsfw = (nsfwDetections, strictness) => {
    if (!nsfwDetections?.length) return false;

    // consider Porn (3), Hentai (1) and Sexy (4) as NSFW, with sexy having stricter threshold above
    const RELEVANT_IDS = new Set([1, 3, 4]);
    const nsfwClasses = getNsfwClasses(strictness);

    const neutral = nsfwDetections.find((d) => d.id === 2); // Neutral
    const drawing = nsfwDetections.find((d) => d.id === 0); // Drawing

    let highestRelevantDelta = 0;

    nsfwDetections.forEach((det) => {
        if (!RELEVANT_IDS.has(det.id)) return;

        const delta = det.probability - nsfwClasses[det.id].thresh;

        if (det.id === 4) {
            // Sexy: must beat Neutral/Drawing by margin to avoid FP
            const sfwMax = Math.max(
                neutral ? neutral.probability : 0,
                drawing ? drawing.probability : 0
            );
            if (delta > 0 && det.probability - sfwMax >= 0.4) {
                highestRelevantDelta = Math.max(highestRelevantDelta, delta);
            }
        } else {
            // Porn or Hentai
            if (delta > 0) {
                highestRelevantDelta = Math.max(highestRelevantDelta, delta);
            }
        }
    });

    return highestRelevantDelta > 0;
};

const genderPredicate = (gender, score, detectMale, detectFemale) => {
    // First, trust the explicit label if it matches requested genders
    if (detectMale && gender === "male") return true;
    if (detectFemale && gender === "female") return true;

    // Fall-back to score heuristics when label is ambiguous
    if (gender === "unknown") return false;

    // In Human, genderScore ~ probability of being female (empirical)
    // Use softer band around 0.5 to decide.
    const femaleProb = score;
    const maleProb = 1 - femaleProb;

    if (detectMale && maleProb >= 0.55) return true;
    if (detectFemale && femaleProb >= 0.55) return true;

    return false;
};

const containsGenderFace = (detections, detectMale, detectFemale) => {
    if (!detections?.face?.length) return false;

    const faces = detections.face;

    // If user asked to blur *both* sexes, accept any confident adult face (>0.6)
    if (detectMale && detectFemale)
        return faces.some(
            (face) =>
                face.age > 16 && (face.score ?? face.confidence ?? 0) > 0.6
        );

    const minAge = detectFemale && !detectMale ? 8 : 16;

    return faces.some((face) => {
        if (face.age <= minAge) return false;
        const conf = face.score ?? face.confidence ?? 0;

        // Special handling when only females should be blurred
        if (detectFemale && !detectMale) {
            // 1. Any unknown-gender adult with reasonably confident face detection
            if (face.gender === "unknown" && conf > 0.6) return true;

            // 2. Faces labelled male but with some female probability (>0.1)
            if (face.gender === "male" && face.genderScore >= 0.1) return true;
        }

        return genderPredicate(
            face.gender,
            face.genderScore,
            detectMale,
            detectFemale
        );
    });
};
// export the human variable and the HUMAN_CONFIG object
export { getNsfwClasses, containsNsfw, containsGenderFace, Detector };
