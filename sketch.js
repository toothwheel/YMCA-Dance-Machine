const URL = "https://teachablemachine.withgoogle.com/models/lTFKiqbhQ/";
let model, webcam, ctx, labelContainer, maxPredictions;

// Variables for floating letters
let letters = [];
let letterColors = {
    'Y': [0, 191, 255], // Electric Blue
    'M': [255, 0, 255], // Pink
    'C': [50, 205, 50], // Lime Green
    'A': [255, 140, 0], // Bright Orange
    'default': [200, 200, 200] // Default gray
};

// Probability threshold for displaying letters (e.g., 0.7 = 70%)
const PROBABILITY_THRESHOLD = 0.7;

// Number of letters to add per pose
const LETTERS_PER_POSE = 5;

// Track the last detected pose
let lastDetectedPose = null;

// Audio element for the YMCA chorus
let ymcaAudio;

async function init() {
    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    // Load the model and metadata
    model = await tmPose.load(modelURL, metadataURL);
    maxPredictions = model.getTotalClasses();

    // Setup webcam
    const size = 400;
    const flip = true; // Whether to flip the webcam
    webcam = new tmPose.Webcam(size, size, flip); // Width, height, flip
    await webcam.setup(); // Request access to the webcam
    await webcam.play();
    window.requestAnimationFrame(loop);

    // Append/get elements to the DOM
    const canvas = document.getElementById("canvas");
    canvas.width = size; canvas.height = size;
    ctx = canvas.getContext("2d");
    labelContainer = document.getElementById("label-container");
    for (let i = 0; i < maxPredictions; i++) { // Add class labels
        labelContainer.appendChild(document.createElement("div"));
    }

    // Initialize the audio element
    ymcaAudio = document.getElementById("ymca-audio");
    ymcaAudio.play(); // Start playing the YMCA chorus
}

async function loop(timestamp) {
    webcam.update(); // Update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // Prediction #1: Run input through PoseNet
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    // Prediction #2: Run input through Teachable Machine classification model
    const prediction = await model.predict(posenetOutput);

    let highestProb = 0;
    let bestClass = "Unknown";

    for (let i = 0; i < maxPredictions; i++) {
        const classPrediction = prediction[i].className + ": " + prediction[i].probability.toFixed(2);
        labelContainer.childNodes[i].innerHTML = classPrediction;

        if (prediction[i].probability > highestProb) {
            highestProb = prediction[i].probability;
            bestClass = prediction[i].className;
        }
    }

    // Log predictions for debugging
    console.log("Best Prediction:", bestClass, "Probability:", highestProb);

    // Check if a new pose is detected
    if (highestProb > PROBABILITY_THRESHOLD && bestClass !== lastDetectedPose) {
        // Clear existing letters when a new pose is detected
        letters = [];
        lastDetectedPose = bestClass; // Update the last detected pose

        // Add multiple floating letters for the detected pose
        for (let i = 0; i < LETTERS_PER_POSE; i++) {
            if (bestClass in letterColors) {
                letters.push(new FloatingLetter(bestClass, letterColors[bestClass]));
            } else {
                letters.push(new FloatingLetter(bestClass, letterColors['default']));
            }
        }
    }

    // Draw the pose and floating letters
    drawPose(pose);
    drawLetters();
}

function drawPose(pose) {
    if (webcam.canvas) {
        ctx.drawImage(webcam.canvas, 0, 0);
        // Draw the keypoints and skeleton
        if (pose) {
            const minPartConfidence = 0.5;
            tmPose.drawKeypoints(pose.keypoints, minPartConfidence, ctx);
            tmPose.drawSkeleton(pose.keypoints, minPartConfidence, ctx);
        }
    }
}

function drawLetters() {
    for (let i = letters.length - 1; i >= 0; i--) {
        let letter = letters[i];
        letter.update();
        letter.display(ctx);
    }
}

// FloatingLetter class
class FloatingLetter {
    constructor(letter, color) {
        this.letter = letter;
        this.color = color;
        this.pos = { x: Math.random() * 400, y: Math.random() * 400 };
        this.vel = { x: (Math.random() - 0.5) * 4, y: (Math.random() - 0.5) * 4 };
        this.size = Math.random() * 20 + 20;
    }

    update() {
        this.pos.x += this.vel.x;
        this.pos.y += this.vel.y;

        // Bounce off edges
        if (this.pos.x < 0 || this.pos.x > 400) this.vel.x *= -1;
        if (this.pos.y < 0 || this.pos.y > 400) this.vel.y *= -1;
    }

    display(ctx) {
        ctx.fillStyle = `rgb(${this.color[0]}, ${this.color[1]}, ${this.color[2]})`;
        ctx.font = `${this.size}px Arial`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(this.letter, this.pos.x, this.pos.y);
    }
}
