/**
 * @file app.js
 * @description Frontend Application Logic for the PureForest Model Verification Sandbox.
 *
 * This module manages interactive UI state, asynchronous communication with the backend
 * REST API, image drag-and-drop ingestion, on-demand tile rendering (True Color RGB & False Color CIR),
 * interactive Confusion Matrix heatmap calculations, Per-Class statistical breakdown,
 * and high-resolution Lightbox inspection.
 *
 * Architecture & Workflows:
 * 1. Overview Dashboard: Fetches and displays territorial split statistics and global benchmark metrics.
 * 2. Accuracy Verifier:
 *    - Drag-and-drop: Base64 encodes multi-channel TIFFs and posts to `/api/predict`.
 *    - Bulk Evaluation: Triggers backend worker threads via `/api/evaluate_test` and polls `/api/job_status`.
 *    - State Caching: Auto-loads last saved evaluation from `/api/last_evaluation`.
 * 3. Analytics Engine: Computes Macro Precision, Recall, F1-Score, and Intersection over Union (IoU).
 */

// ----------------- Global Application State -----------------
/** @type {Object|null} Cached metadata and dataset partition statistics */
let appStats = null;

/** @type {Array<Object>} List of prediction result objects currently evaluated */
let resultsList = [];

/** @type {Array<File>} In-memory queue of files selected for drag-and-drop inference */
let uploadQueue = [];

/** @type {number} Current index pointer in uploadQueue */
let currentUploadIndex = 0;

/** @type {boolean} Flag indicating whether an active upload or evaluation job is running */
let isUploading = false;

/** @type {string} Current image channel visualization mode ('true' for RGB, 'false' for CIR) */
let currentImageView = 'true';

/** @type {number|null} Interval ID for polling asynchronous evaluation jobs */
let pollIntervalId = null;

/** @type {Object<number, string>} Semantic class mapping matching backend IDs */
const CLASS_MAPPING = {
    0: "Deciduous oak",
    1: "Evergreen oak",
    2: "Beech",
    3: "Chestnut",
    4: "Black locust",
    5: "Maritime pine",
    6: "Scotch pine",
    7: "Black pine",
    8: "Aleppo pine",
    9: "Fir",
    10: "Spruce",
    11: "Larch",
    12: "Douglas"
};

// ----------------- DOM Element References -----------------
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
const splitsTableBody = document.getElementById('splits-table-body');
const classesTableBody = document.getElementById('classes-table-body');

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const progressContainer = document.getElementById('progress-container');
const progressLabel = document.getElementById('progress-label');
const progressPercentage = document.getElementById('progress-percentage');
const progressBarFill = document.getElementById('progress-bar-fill');
const clearBtn = document.getElementById('clear-btn');

const runBulkTestBtn = document.getElementById('run-bulk-test-btn');
const bulkTestLimit = document.getElementById('bulk-test-limit');
const bulkProgressContainer = document.getElementById('bulk-progress-container');
const bulkProgressLabel = document.getElementById('bulk-progress-label');
const bulkProgressPercentage = document.getElementById('bulk-progress-percentage');
const bulkProgressBarFill = document.getElementById('bulk-progress-bar-fill');

const overviewModelName = document.getElementById('overview-model-name');
const overviewEvaluateBtn = document.getElementById('overview-evaluate-btn');
const overviewAccuracy = document.getElementById('overview-accuracy');
const overviewF1 = document.getElementById('overview-f1');
const overviewPrecision = document.getElementById('overview-precision');
const overviewRecall = document.getElementById('overview-recall');
const overviewProgressContainer = document.getElementById('overview-progress-container');
const overviewProgressLabel = document.getElementById('overview-progress-label');
const overviewProgressPercentage = document.getElementById('overview-progress-percentage');
const overviewProgressBarFill = document.getElementById('overview-progress-bar-fill');

const sessionAccuracy = document.getElementById('session-accuracy');
const sessionTotal = document.getElementById('session-total');
const sessionCorrect = document.getElementById('session-correct');

const welcomePanel = document.getElementById('welcome-panel');
const resultsPanel = document.getElementById('results-panel');

const btnTrueColor = document.getElementById('btn-true-color');
const btnFalseColor = document.getElementById('btn-false-color');

const subnavBtns = document.querySelectorAll('.subnav-btn');
const subtabContents = document.querySelectorAll('.subtab-content');

const imageGallery = document.getElementById('image-gallery');
const gallerySearch = document.getElementById('gallery-search');
const pills = document.querySelectorAll('.pill');
const pillAllCount = document.getElementById('pill-all-count');
const pillCorrectCount = document.getElementById('pill-correct-count');
const pillIncorrectCount = document.getElementById('pill-incorrect-count');

const matrixContainer = document.getElementById('matrix-container');
const perclassTableBody = document.getElementById('perclass-table-body');
const misclassifiedTableBody = document.getElementById('misclassified-table-body');

const lightbox = document.getElementById('lightbox');
const lightboxClose = document.querySelector('.lightbox-close');
const lightboxImgTrue = document.getElementById('lightbox-img-true');
const lightboxImgFalse = document.getElementById('lightbox-img-false');
const lightboxFilename = document.getElementById('lightbox-filename');
const lightboxSpecies = document.getElementById('lightbox-species');
const lightboxTrue = document.getElementById('lightbox-true');
const lightboxPred = document.getElementById('lightbox-pred');
const lightboxStatus = document.getElementById('lightbox-status');

/** Base URL for API requests (empty for relative path on same host) */
const API_HOST = '';

// ----------------- Initialization -----------------
document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    setupNavigation();
    setupDragAndDrop();
    setupGalleryControls();
    setupLightbox();
    setupBulkEvaluation();
    setupOverviewEvaluation();
    loadCachedEvaluation();
});

// ----------------- Navigation & Tab Routing -----------------
/**
 * Attaches event listeners for primary top navigation and analytical sub-tabs.
 */
function setupNavigation() {
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        });
    });

    subnavBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const subtabId = btn.getAttribute('data-subtab');
            subnavBtns.forEach(b => b.classList.remove('active'));
            subtabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(subtabId).classList.add('active');

            if (subtabId === 'matrix-tab') {
                renderConfusionMatrix();
            } else if (subtabId === 'perclass-tab') {
                renderPerClassMetrics();
            } else if (subtabId === 'misclassified-tab') {
                renderMisclassifiedList();
            }
        });
    });
}

// ----------------- Dataset Statistics -----------------
/**
 * Asynchronously retrieves dataset partition metadata and active model type from `/api/stats`.
 */
async function fetchStats() {
    try {
        const res = await fetch(`${API_HOST}/api/stats`);
        if (!res.ok) throw new Error('Stats API failed');
        appStats = await res.json();

        populateStatsTables();

        const badge = document.getElementById('active-model-badge');
        if (badge && appStats.loaded_model) {
            badge.textContent = `Active Model: ${appStats.loaded_model}`;
            badge.style.display = 'inline-block';
        }
        if (overviewModelName && appStats.loaded_model) {
            overviewModelName.textContent = appStats.loaded_model;
        }
    } catch (err) {
        console.error('Error fetching statistics:', err);
    }
}

/**
 * Populates HTML summary tables in the Overview tab with territorial splits and class statistics.
 */
function populateStatsTables() {
    if (!appStats) return;

    splitsTableBody.innerHTML = '';
    Object.keys(appStats.splits).forEach(splitName => {
        const split = appStats.splits[splitName];
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight: 600;">${splitName}</td>
            <td>${split.area_km2.toFixed(2)}</td>
            <td>${split.patches.toLocaleString()}</td>
            <td>${((split.patches / 135569) * 100).toFixed(1)}%</td>
            <td>${split.polygons}</td>
        `;
        splitsTableBody.appendChild(tr);
    });

    classesTableBody.innerHTML = '';
    appStats.classes.forEach(cls => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><code>${cls.id}</code></td>
            <td style="font-weight: 500; color: #fff;">${cls.name}</td>
            <td style="font-style: italic; font-size: 0.8rem;">${cls.species.join(', ')}</td>
            <td>${cls.train.toLocaleString()}</td>
            <td>${cls.val.toLocaleString()}</td>
            <td>${cls.test.toLocaleString()}</td>
            <td style="font-weight: 600; color: var(--primary);">${(cls.train + cls.val + cls.test).toLocaleString()}</td>
        `;
        classesTableBody.appendChild(tr);
    });
}

// ----------------- Drag & Drop Upload & Ingestion -----------------
/**
 * Sets up drag-and-drop event listeners and file browser triggers.
 */
function setupDragAndDrop() {
    dropzone.addEventListener('click', () => {
        if (!isUploading) fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        handleSelectedFiles(e.target.files);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        handleSelectedFiles(dt.files);
    }, false);

    clearBtn.addEventListener('click', () => {
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
        }

        resultsList = [];
        uploadQueue = [];
        currentUploadIndex = 0;
        isUploading = false;

        progressContainer.style.display = 'none';
        bulkProgressContainer.style.display = 'none';
        welcomePanel.style.display = 'flex';
        resultsPanel.style.display = 'none';
        clearBtn.disabled = true;
        runBulkTestBtn.disabled = false;

        updateMetricsDisplay();
        imageGallery.innerHTML = '';
        misclassifiedTableBody.innerHTML = '';
    });
}

/**
 * Filters uploaded files for TIFF extensions and initiates the sequential upload queue.
 * @param {FileList|Array<File>} files - Ingested files from drop event or file input.
 */
function handleSelectedFiles(files) {
    if (isUploading) return;

    const tiffFiles = Array.from(files).filter(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        return ext === 'tiff' || ext === 'tif';
    });

    if (tiffFiles.length === 0) {
        alert('Please drop valid .tiff or .tif files from the PureForest dataset.');
        return;
    }

    uploadQueue = tiffFiles;
    currentUploadIndex = 0;
    isUploading = true;

    progressContainer.style.display = 'block';
    clearBtn.disabled = true;
    runBulkTestBtn.disabled = true;

    welcomePanel.style.display = 'none';
    resultsPanel.style.display = 'block';

    subnavBtns[0].click();
    processNextFile();
}

/**
 * Recursively uploads and predicts the next TIFF file in the uploadQueue.
 */
function processNextFile() {
    if (currentUploadIndex >= uploadQueue.length) {
        isUploading = false;
        progressLabel.textContent = 'Verification Complete';
        clearBtn.disabled = false;
        runBulkTestBtn.disabled = false;
        return;
    }

    const file = uploadQueue[currentUploadIndex];
    const percent = Math.round((currentUploadIndex / uploadQueue.length) * 100);
    progressLabel.textContent = `Processing file ${currentUploadIndex + 1} of ${uploadQueue.length}...`;
    progressPercentage.textContent = `${percent}%`;
    progressBarFill.style.width = `${percent}%`;

    const reader = new FileReader();
    reader.onload = async (event) => {
        const base64Content = event.target.result.split(',')[1];

        try {
            const response = await fetch(`${API_HOST}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: file.name,
                    content: base64Content
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error during classification');
            }

            const result = await response.json();
            resultsList.push(result);

            appendImageCard(result);
            updateMetricsDisplay();

        } catch (err) {
            console.error(`Failed to analyze ${file.name}:`, err);
            appendErrorCard(file.name, err.message);
        } finally {
            currentUploadIndex++;
            setTimeout(processNextFile, 50);
        }
    };

    reader.readAsDataURL(file);
}

// ----------------- Bulk Dataset Evaluation -----------------
/**
 * Starts a background asynchronous evaluation job on server-side dataset splits.
 * @param {number} limitVal - Number of samples to evaluate (0 for full split).
 * @param {string} splitVal - Target dataset split ('test', 'train', 'val').
 */
async function executeBulkEvaluation(limitVal, splitVal = 'test') {
    if (isUploading) return;
    isUploading = true;

    clearBtn.disabled = true;
    runBulkTestBtn.disabled = true;
    if (overviewEvaluateBtn) {
        overviewEvaluateBtn.disabled = true;
        overviewEvaluateBtn.textContent = 'Evaluating...';
    }

    bulkProgressContainer.style.display = 'block';
    bulkProgressLabel.textContent = 'Initializing background evaluation job...';
    bulkProgressPercentage.textContent = '0%';
    bulkProgressBarFill.style.width = '0%';

    if (overviewProgressContainer) {
        overviewProgressContainer.style.display = 'block';
        overviewProgressLabel.textContent = 'Initializing background evaluation job...';
        overviewProgressPercentage.textContent = '0%';
        overviewProgressBarFill.style.width = '0%';
    }

    welcomePanel.style.display = 'none';
    resultsPanel.style.display = 'block';
    imageGallery.innerHTML = '';
    imageGallery.innerHTML = '<div style="grid-column: 1/-1; padding: 2rem; text-align: center; color: var(--text-muted);"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="tree-icon" style="width:3rem;height:3rem;margin-bottom:1rem;animation:pulse 2s infinite;"><path d="M12 2v20M17 5H7M19 12H5M21 19H3"/></svg><br>Background evaluation running... Check progress bar.</div>';

    try {
        const response = await fetch(`${API_HOST}/api/evaluate_test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ limit: limitVal, split: splitVal })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Failed to start bulk evaluation');
        }

        const jobData = await response.json();
        const jobId = jobData.job_id;
        const totalToProcess = jobData.total;

        pollIntervalId = setInterval(() => pollJobStatus(jobId, totalToProcess), 1000);

    } catch (err) {
        alert(`Error starting bulk evaluation: ${err.message}`);
        imageGallery.innerHTML = `<div style="grid-column: 1/-1; padding: 2rem; text-align: center; color: var(--error);">Error: ${err.message}</div>`;
        welcomePanel.style.display = 'flex';
        resultsPanel.style.display = 'none';
        isUploading = false;
        clearBtn.disabled = false;
        runBulkTestBtn.disabled = false;
        if (overviewEvaluateBtn) {
            overviewEvaluateBtn.disabled = false;
            overviewEvaluateBtn.textContent = 'Run Test Evaluation';
        }
        bulkProgressContainer.style.display = 'none';
        if (overviewProgressContainer) overviewProgressContainer.style.display = 'none';
    }
}

/**
 * Initializes button listeners for Bulk Evaluation panel.
 */
function setupBulkEvaluation() {
    runBulkTestBtn.addEventListener('click', () => {
        const limitVal = parseInt(bulkTestLimit.value, 10);
        const splitSelect = document.getElementById('bulk-test-split');
        const splitVal = splitSelect ? splitSelect.value : 'test';
        executeBulkEvaluation(limitVal, splitVal);
    });
}

/**
 * Initializes button listeners for Overview tab evaluation trigger.
 */
function setupOverviewEvaluation() {
    if (overviewEvaluateBtn) {
        overviewEvaluateBtn.addEventListener('click', () => {
            const limitVal = parseInt(bulkTestLimit.value, 10);
            const splitSelect = document.getElementById('bulk-test-split');
            const splitVal = splitSelect ? splitSelect.value : 'test';
            executeBulkEvaluation(limitVal, splitVal);
        });
    }
}

/**
 * Polls the `/api/job_status` endpoint every second until the job completes or fails.
 * @param {string} jobId - UUID of the running evaluation job.
 * @param {number} total - Total number of image patches being evaluated.
 */
async function pollJobStatus(jobId, total) {
    try {
        const res = await fetch(`${API_HOST}/api/job_status?job_id=${jobId}`);
        if (!res.ok) throw new Error('Failed to query job status');

        const job = await res.json();

        if (job.status === 'running') {
            const processed = job.processed;
            const percent = Math.round((processed / total) * 100);
            const text = `Evaluating patches: ${processed.toLocaleString()} / ${total.toLocaleString()}...`;

            bulkProgressLabel.textContent = text;
            bulkProgressPercentage.textContent = `${percent}%`;
            bulkProgressBarFill.style.width = `${percent}%`;

            if (overviewProgressLabel) overviewProgressLabel.textContent = text;
            if (overviewProgressPercentage) overviewProgressPercentage.textContent = `${percent}%`;
            if (overviewProgressBarFill) overviewProgressBarFill.style.width = `${percent}%`;
        }
        else if (job.status === 'completed') {
            clearInterval(pollIntervalId);
            pollIntervalId = null;

            const finishText = 'Formatting analytics charts...';
            bulkProgressLabel.textContent = finishText;
            bulkProgressPercentage.textContent = '95%';
            bulkProgressBarFill.style.width = '95%';

            if (overviewProgressLabel) overviewProgressLabel.textContent = finishText;
            if (overviewProgressPercentage) overviewProgressPercentage.textContent = '95%';
            if (overviewProgressBarFill) overviewProgressBarFill.style.width = '95%';

            resultsList = job.result.predictions;

            imageGallery.innerHTML = '';
            const renderLimit = Math.min(resultsList.length, 100);
            for (let i = 0; i < renderLimit; i++) {
                appendImageCard(resultsList[i]);
            }

            if (resultsList.length > 100) {
                const note = document.createElement('div');
                note.style.gridColumn = '1 / -1';
                note.style.padding = '1rem';
                note.style.background = 'rgba(255,255,255,0.03)';
                note.style.borderRadius = '8px';
                note.style.textAlign = 'center';
                note.style.fontSize = '0.85rem';
                note.style.color = 'var(--text-muted)';
                note.innerHTML = `Showing first 100 image thumbnails of ${resultsList.length.toLocaleString()} total to maintain page speed. The metrics tabs contain full evaluation analytics.`;
                imageGallery.insertBefore(note, imageGallery.firstChild);
            }

            updateMetricsDisplay();

            const activeSubtab = document.querySelector('.subnav-btn.active').getAttribute('data-subtab');
            if (activeSubtab === 'matrix-tab') {
                renderConfusionMatrix();
            } else if (activeSubtab === 'perclass-tab') {
                renderPerClassMetrics();
            } else if (activeSubtab === 'misclassified-tab') {
                renderMisclassifiedList();
            }

            bulkProgressLabel.textContent = 'Bulk Evaluation Complete';
            bulkProgressPercentage.textContent = '100%';
            bulkProgressBarFill.style.width = '100%';

            if (overviewProgressLabel) overviewProgressLabel.textContent = 'Bulk Evaluation Complete';
            if (overviewProgressPercentage) overviewProgressPercentage.textContent = '100%';
            if (overviewProgressBarFill) overviewProgressBarFill.style.width = '100%';

            setTimeout(() => {
                bulkProgressContainer.style.display = 'none';
                if (overviewProgressContainer) overviewProgressContainer.style.display = 'none';
            }, 2000);

            isUploading = false;
            clearBtn.disabled = false;
            runBulkTestBtn.disabled = false;
            if (overviewEvaluateBtn) {
                overviewEvaluateBtn.disabled = false;
                overviewEvaluateBtn.textContent = 'Run Test Evaluation';
            }
        }
        else if (job.status === 'failed') {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
            bulkProgressContainer.style.display = 'none';
            if (overviewProgressContainer) overviewProgressContainer.style.display = 'none';
            if (overviewEvaluateBtn) {
                overviewEvaluateBtn.disabled = false;
                overviewEvaluateBtn.textContent = 'Run Test Evaluation';
            }
            throw new Error(job.error || 'Internal background job error');
        }
    } catch (err) {
        clearInterval(pollIntervalId);
        pollIntervalId = null;
        alert(`Bulk Evaluation Job Error: ${err.message}`);
        imageGallery.innerHTML = `<div style="grid-column: 1/-1; padding: 2rem; text-align: center; color: var(--error);">Error: ${err.message}</div>`;
        welcomePanel.style.display = 'flex';
        resultsPanel.style.display = 'none';
        isUploading = false;
        clearBtn.disabled = false;
        runBulkTestBtn.disabled = false;
        if (overviewEvaluateBtn) {
            overviewEvaluateBtn.disabled = false;
            overviewEvaluateBtn.textContent = 'Run Test Evaluation';
        }
        bulkProgressContainer.style.display = 'none';
        if (overviewProgressContainer) overviewProgressContainer.style.display = 'none';
    }
}

// ----------------- Metrics & Analytics Computation -----------------
/**
 * Updates all session indicators, count badges, and macro statistical cards.
 */
function updateMetricsDisplay() {
    const total = resultsList.length;
    const correct = resultsList.filter(r => r.correct === true).length;

    sessionTotal.textContent = total.toLocaleString();
    sessionCorrect.textContent = correct.toLocaleString();

    if (total > 0) {
        const acc = (correct / total) * 100;
        sessionAccuracy.textContent = `${acc.toFixed(1)}%`;
    } else {
        sessionAccuracy.textContent = '--';
    }

    pillAllCount.textContent = total.toLocaleString();
    pillCorrectCount.textContent = correct.toLocaleString();
    pillIncorrectCount.textContent = (total - correct).toLocaleString();

    if (total > 0) {
        const metrics = calculateMacroMetrics(resultsList);
        if (overviewAccuracy) overviewAccuracy.textContent = `${metrics.accuracy.toFixed(1)}%`;
        if (overviewF1) overviewF1.textContent = `${metrics.f1.toFixed(1)}%`;
        if (overviewPrecision) overviewPrecision.textContent = `${metrics.precision.toFixed(1)}%`;
        if (overviewRecall) overviewRecall.textContent = `${metrics.recall.toFixed(1)}%`;
    } else {
        if (overviewAccuracy) overviewAccuracy.textContent = '--';
        if (overviewF1) overviewF1.textContent = '--';
        if (overviewPrecision) overviewPrecision.textContent = '--';
        if (overviewRecall) overviewRecall.textContent = '--';
    }
}

// ----------------- Gallery & Rendering Controls -----------------
/**
 * Configures view toggles (True Color vs CIR), live search input, and filter pills.
 */
function setupGalleryControls() {
    btnTrueColor.addEventListener('click', () => {
        btnTrueColor.classList.add('active');
        btnFalseColor.classList.remove('active');
        currentImageView = 'true';
        updateGalleryImageSources();
    });

    btnFalseColor.addEventListener('click', () => {
        btnFalseColor.classList.add('active');
        btnTrueColor.classList.remove('active');
        currentImageView = 'false';
        updateGalleryImageSources();
    });

    gallerySearch.addEventListener('input', filterGallery);

    pills.forEach(p => {
        p.addEventListener('click', () => {
            pills.forEach(x => x.classList.remove('active'));
            p.classList.add('active');
            filterGallery();
        });
    });
}

/**
 * Switches the active image source of gallery cards between RGB and CIR views.
 */
function updateGalleryImageSources() {
    const cards = imageGallery.querySelectorAll('.image-card');
    cards.forEach(card => {
        const img = card.querySelector('img');
        const tc = card.getAttribute('data-true-color');
        const fc = card.getAttribute('data-false-color');
        if (img && tc && fc) {
            img.src = currentImageView === 'true' ? tc : fc;
        }
    });
}

/**
 * Filters visible gallery cards by query string and correct/incorrect status.
 */
function filterGallery() {
    const query = gallerySearch.value.toLowerCase();
    const activePill = document.querySelector('.pill.active').getAttribute('data-filter');
    const cards = imageGallery.querySelectorAll('.image-card');

    cards.forEach(card => {
        const species = card.getAttribute('data-species') ? card.getAttribute('data-species').toLowerCase() : '';
        const trueClass = card.getAttribute('data-true-class') ? card.getAttribute('data-true-class').toLowerCase() : '';
        const predClass = card.getAttribute('data-pred-class') ? card.getAttribute('data-pred-class').toLowerCase() : '';
        const correct = card.getAttribute('data-correct') === 'true';

        const matchesSearch = species.includes(query) || trueClass.includes(query) || predClass.includes(query);
        let matchesPill = true;

        if (activePill === 'correct') {
            matchesPill = correct;
        } else if (activePill === 'incorrect') {
            matchesPill = !correct;
        }

        if (matchesSearch && matchesPill) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

/**
 * Generates and appends an interactive card element to the image gallery.
 * @param {Object} result - Prediction result object from API.
 */
function appendImageCard(result) {
    const card = document.createElement('div');
    card.className = `image-card ${result.correct ? 'correct' : 'incorrect'}`;
    card.setAttribute('data-species', result.true_species);
    card.setAttribute('data-true-class', result.true_class_name);
    card.setAttribute('data-pred-class', result.predicted_class_name);
    card.setAttribute('data-correct', result.correct);

    const trueColorSrc = result.true_color_image ? `data:image/png;base64,${result.true_color_image}` : '';
    const falseColorSrc = result.false_color_image ? `data:image/png;base64,${result.false_color_image}` : '';

    card.setAttribute('data-true-color', trueColorSrc);
    card.setAttribute('data-false-color', falseColorSrc);

    const placeholder = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width: 2.25rem; color: var(--border-color-active);"><path d="M12 22V5M12 5L7 10M12 5L17 10M19 14l-7-7-7 7M21 19H3"/></svg>';

    card.innerHTML = `
        <div class="card-img-wrapper" style="display: flex; align-items: center; justify-content: center; background: #0b0f0d; overflow: hidden; height: 160px; position: relative;">
            ${trueColorSrc ? `<img src="${currentImageView === 'true' ? trueColorSrc : falseColorSrc}" alt="${result.filename}" style="width:100%;height:100%;object-fit:cover;">` : placeholder}
            <span class="card-status-badge ${result.correct ? 'correct' : 'incorrect'}" style="position:absolute;top:8px;right:8px;">
                ${result.correct ? 'Correct' : 'Incorrect'}
            </span>
        </div>
        <div class="card-details" style="padding: 10px;">
            <h4 class="card-species" title="${result.true_species}" style="font-size:0.82rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#fff;">${result.true_species}</h4>
            <p class="card-pred" style="font-size:0.7rem;color:var(--text-muted);margin-top:2px;">Pred: <span style="font-weight:600;color:${result.correct ? 'var(--success)' : 'var(--error)'};">${result.predicted_class_name}</span></p>
        </div>
    `;

    card.addEventListener('click', () => openLightbox(card, result));
    imageGallery.appendChild(card);
}

/**
 * Appends an error placeholder card for invalid or corrupted image uploads.
 * @param {string} filename - Name of the failed file.
 * @param {string} errorMsg - Error description.
 */
function appendErrorCard(filename, errorMsg) {
    const card = document.createElement('div');
    card.className = 'image-card incorrect';
    card.innerHTML = `
        <div class="card-img-wrapper" style="display:flex;align-items:center;justify-content:center;background:#1a1010;color:var(--error);padding:1rem;font-size:0.75rem;text-align:center;">
            Failed to parse: ${errorMsg}
        </div>
        <div class="card-details">
            <h4 class="card-species" style="color:var(--error);">${filename}</h4>
            <p class="card-pred">Classification failed</p>
        </div>
    `;
    imageGallery.appendChild(card);
}

// ----------------- Lightbox Modal -----------------
/**
 * Sets up modal backdrop click and close button listeners for the Lightbox.
 */
function setupLightbox() {
    lightboxClose.addEventListener('click', () => {
        lightbox.style.display = 'none';
    });
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) lightbox.style.display = 'none';
    });
}

/**
 * Opens high-resolution True Color & False Color modal inspection view.
 * If images are not pre-rendered in memory, fetches base64 PNGs on demand from `/api/image`.
 * @param {HTMLElement|null} cardElement - Related thumbnail card element.
 * @param {Object} result - Prediction result object.
 */
async function openLightbox(cardElement, result) {
    let trueColorSrc = cardElement ? cardElement.getAttribute('data-true-color') : '';
    let falseColorSrc = cardElement ? cardElement.getAttribute('data-false-color') : '';

    if (!trueColorSrc) {
        lightboxFilename.textContent = "Loading image channels...";
        lightboxImgTrue.src = '';
        lightboxImgFalse.src = '';
        lightbox.style.display = 'flex';

        try {
            const res = await fetch(`${API_HOST}/api/image?filename=${result.filename}`);
            if (!res.ok) throw new Error('Image fetch from server failed');
            const data = await res.json();

            trueColorSrc = `data:image/png;base64,${data.true_color_image}`;
            falseColorSrc = `data:image/png;base64,${data.false_color_image}`;

            if (cardElement) {
                cardElement.setAttribute('data-true-color', trueColorSrc);
                cardElement.setAttribute('data-false-color', falseColorSrc);
                const wrapper = cardElement.querySelector('.card-img-wrapper');
                wrapper.innerHTML = `<img src="${currentImageView === 'true' ? trueColorSrc : falseColorSrc}" alt="${result.filename}" style="width:100%;height:100%;object-fit:cover;">
                <span class="card-status-badge ${result.correct ? 'correct' : 'incorrect'}" style="position:absolute;top:8px;right:8px;">
                    ${result.correct ? 'Correct' : 'Incorrect'}
                </span>`;
            }
        } catch (err) {
            console.error(err);
            lightbox.style.display = 'none';
            alert(`Error loading image from local disk: ${err.message}`);
            return;
        }
    }

    lightboxFilename.textContent = result.filename;
    lightboxSpecies.textContent = result.true_species;
    lightboxTrue.textContent = `${result.true_class_name} (Class ${result.true_class_id})`;
    lightboxPred.textContent = `${result.predicted_class_name} (Class ${result.predicted_class_id})`;

    lightboxStatus.textContent = result.correct ? 'Correct' : 'Incorrect';
    lightboxStatus.className = `value badge ${result.correct ? 'correct' : 'incorrect'}`;

    lightboxImgTrue.src = trueColorSrc;
    lightboxImgFalse.src = falseColorSrc;

    lightbox.style.display = 'flex';
}

// ----------------- Confusion Matrix Heatmap -----------------
/**
 * Renders the interactive 13x13 Confusion Matrix heatmap grid with normalized row ratios.
 */
function renderConfusionMatrix() {
    matrixContainer.innerHTML = '';
    const size = 13;

    matrixContainer.style.gridTemplateColumns = `120px repeat(${size}, 1fr)`;

    const labelEmpty = document.createElement('div');
    labelEmpty.className = 'matrix-header-cell';
    labelEmpty.style.background = 'rgba(0,0,0,0.5)';
    labelEmpty.innerHTML = 'GT \\ Pred';
    matrixContainer.appendChild(labelEmpty);

    for (let c = 0; c < size; c++) {
        const colCell = document.createElement('div');
        colCell.className = 'matrix-header-cell';
        colCell.title = CLASS_MAPPING[c];
        colCell.innerHTML = `C${c}`;
        matrixContainer.appendChild(colCell);
    }

    const matrix = Array.from({ length: size }, () => Array(size).fill(0));
    resultsList.forEach(r => {
        if (r.true_class_id !== null && r.true_class_id >= 0 && r.true_class_id < size &&
            r.predicted_class_id !== null && r.predicted_class_id >= 0 && r.predicted_class_id < size) {
            matrix[r.true_class_id][r.predicted_class_id]++;
        }
    });

    const rowMaxes = matrix.map(row => Math.max(...row, 1));

    for (let r = 0; r < size; r++) {
        const rowLabel = document.createElement('div');
        rowLabel.className = 'matrix-row-label';
        rowLabel.title = CLASS_MAPPING[r];
        rowLabel.innerHTML = `C${r} ${CLASS_MAPPING[r]}`;
        matrixContainer.appendChild(rowLabel);

        for (let c = 0; c < size; c++) {
            const count = matrix[r][c];
            const cell = document.createElement('div');
            cell.className = `matrix-cell ${count > 0 ? 'has-val' : ''} ${r === c ? 'diagonal' : ''}`;

            if (count > 0) {
                cell.innerHTML = count.toLocaleString();
                const ratio = count / rowMaxes[r];
                cell.style.backgroundColor = `rgba(16, 185, 129, ${0.15 + ratio * 0.8})`;
            } else {
                cell.innerHTML = '0';
                cell.style.backgroundColor = 'rgba(0,0,0,0.15)';
            }

            const tooltip = document.createElement('div');
            tooltip.className = 'cell-tooltip';
            tooltip.innerHTML = `
                <strong>GT:</strong> ${CLASS_MAPPING[r]} (C${r})<br>
                <strong>Pred:</strong> ${CLASS_MAPPING[c]} (C${c})<br>
                <strong>Count:</strong> ${count.toLocaleString()} patches
            `;
            cell.appendChild(tooltip);
            matrixContainer.appendChild(cell);
        }
    }
}

// ----------------- Per-Class Metrics Table -----------------
/**
 * Computes and renders True Positives, True Negatives, False Positives, False Negatives,
 * Precision, Recall, F1-Score, and Intersection over Union (IoU) per class.
 */
function renderPerClassMetrics() {
    perclassTableBody.innerHTML = '';
    const size = 13;

    const stats = Array.from({ length: size }, (_, i) => ({
        id: i,
        name: CLASS_MAPPING[i],
        tp: 0,
        tn: 0,
        fp: 0,
        fn: 0,
        support: 0
    }));

    resultsList.forEach(r => {
        const trueId = r.true_class_id;
        const predId = r.predicted_class_id;

        if (trueId !== null && trueId >= 0 && trueId < size) {
            stats[trueId].support++;
        }

        for (let i = 0; i < size; i++) {
            if (trueId === i && predId === i) {
                stats[i].tp++;
            } else if (trueId === i && predId !== i) {
                stats[i].fn++;
            } else if (trueId !== i && predId === i) {
                stats[i].fp++;
            } else if (trueId !== i && predId !== i) {
                stats[i].tn++;
            }
        }
    });

    stats.forEach(item => {
        const tr = document.createElement('tr');

        let precision = 0;
        let recall = 0;
        let f1 = 0;
        let iou = 0;

        if (item.tp + item.fp > 0) {
            precision = (item.tp / (item.tp + item.fp)) * 100;
        }
        if (item.tp + item.fn > 0) {
            recall = (item.tp / (item.tp + item.fn)) * 100;
        }
        if (precision + recall > 0) {
            f1 = (2 * precision * recall) / (precision + recall);
        }
        if (item.tp + item.fp + item.fn > 0) {
            iou = (item.tp / (item.tp + item.fp + item.fn)) * 100;
        }

        tr.innerHTML = `
            <td style="font-weight:600;color:#fff;">${item.name}</td>
            <td>${item.support.toLocaleString()}</td>
            <td>${item.tp.toLocaleString()}</td>
            <td>${item.tn.toLocaleString()}</td>
            <td>${item.fp.toLocaleString()}</td>
            <td>${item.fn.toLocaleString()}</td>
            <td>${item.support > 0 || item.tp + item.fp > 0 ? precision.toFixed(1) + '%' : '--'}</td>
            <td>${item.support > 0 ? recall.toFixed(1) + '%' : '--'}</td>
            <td>${precision + recall > 0 ? f1.toFixed(1) + '%' : '--'}</td>
            <td>${item.tp + item.fp + item.fn > 0 ? iou.toFixed(1) + '%' : '--'}</td>
        `;
        perclassTableBody.appendChild(tr);
    });
}

// ----------------- Misclassified Instances -----------------
/**
 * Renders the table of misclassified instances with quick-access Lightbox preview buttons.
 */
function renderMisclassifiedList() {
    misclassifiedTableBody.innerHTML = '';
    const misclassified = resultsList.filter(r => r.correct === false);

    if (misclassified.length === 0) {
        misclassifiedTableBody.innerHTML = `
            <tr>
                <td colspan="4" style="text-align: center; color: var(--success); font-weight: 500; padding: 2rem;">
                    No misclassified instances! Accuracy is 100%.
                </td>
            </tr>
        `;
        return;
    }

    misclassified.forEach(item => {
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.innerHTML = `
            <td style="font-family: monospace; font-size: 0.8rem;">${item.filename}</td>
            <td style="font-weight: 500; color: #fff;">${item.true_species} (${item.true_class_name})</td>
            <td style="color: var(--error); font-weight: 500;">${item.predicted_class_name}</td>
            <td style="text-align: center;">
                <button class="pill" style="padding: 0.25rem 0.65rem; background: var(--error-bg); border-color: var(--error); color: var(--error); cursor: pointer; outline: none;">
                    View Image
                </button>
            </td>
        `;

        tr.addEventListener('click', () => {
            const cards = imageGallery.querySelectorAll('.image-card');
            let matchedCard = null;
            for (let card of cards) {
                const img = card.querySelector('img');
                if (img && img.alt === item.filename) {
                    matchedCard = card;
                    break;
                }
            }
            openLightbox(matchedCard, item);
        });

        misclassifiedTableBody.appendChild(tr);
    });
}

// ----------------- Macro Statistical Calculator -----------------
/**
 * Calculates unweighted macro-averaged precision, recall, F1, and overall accuracy.
 * @param {Array<Object>} predictions - Full prediction list.
 * @returns {Object} Calculated metrics object {accuracy, precision, recall, f1}.
 */
function calculateMacroMetrics(predictions) {
    const size = 13;
    const stats = Array.from({ length: size }, () => ({ tp: 0, fp: 0, fn: 0, support: 0 }));

    predictions.forEach(r => {
        const trueId = r.true_class_id;
        const predId = r.predicted_class_id;
        if (trueId !== null && trueId >= 0 && trueId < size) {
            stats[trueId].support++;
        }
        for (let i = 0; i < size; i++) {
            if (trueId === i && predId === i) {
                stats[i].tp++;
            } else if (trueId === i && predId !== i) {
                stats[i].fn++;
            } else if (trueId !== i && predId === i) {
                stats[i].fp++;
            }
        }
    });

    let totalPrecision = 0;
    let totalRecall = 0;
    let totalF1 = 0;
    let validPrecisionClasses = 0;
    let validRecallClasses = 0;
    let validF1Classes = 0;

    stats.forEach(item => {
        let precision = 0;
        let recall = 0;
        let f1 = 0;

        if (item.tp + item.fp > 0) {
            precision = item.tp / (item.tp + item.fp);
            totalPrecision += precision;
            validPrecisionClasses++;
        }
        if (item.tp + item.fn > 0) {
            recall = item.tp / (item.tp + item.fn);
            totalRecall += recall;
            validRecallClasses++;
        }
        if (precision + recall > 0) {
            f1 = (2 * precision * recall) / (precision + recall);
            totalF1 += f1;
            validF1Classes++;
        }
    });

    const macroPrecision = validPrecisionClasses > 0 ? (totalPrecision / validPrecisionClasses) * 100 : 0;
    const macroRecall = validRecallClasses > 0 ? (totalRecall / validRecallClasses) * 100 : 0;
    const macroF1 = validF1Classes > 0 ? (totalF1 / validF1Classes) * 100 : 0;

    const correct = predictions.filter(r => r.correct === true).length;
    const accuracy = predictions.length > 0 ? (correct / predictions.length) * 100 : 0;

    return {
        accuracy: accuracy,
        precision: macroPrecision,
        recall: macroRecall,
        f1: macroF1
    };
}

// ----------------- Cache Restoration -----------------
/**
 * Automatically fetches the last cached evaluation from `/api/last_evaluation` on initial page load.
 */
async function loadCachedEvaluation() {
    try {
        const response = await fetch(`${API_HOST}/api/last_evaluation`);
        if (!response.ok) {
            return;
        }

        const cached = await response.json();
        if (cached && cached.predictions && cached.predictions.length > 0) {
            resultsList = cached.predictions;

            welcomePanel.style.display = 'none';
            resultsPanel.style.display = 'block';
            clearBtn.disabled = false;

            imageGallery.innerHTML = '';
            const renderLimit = Math.min(resultsList.length, 100);
            for (let i = 0; i < renderLimit; i++) {
                appendImageCard(resultsList[i]);
            }

            if (resultsList.length > 100) {
                const note = document.createElement('div');
                note.style.gridColumn = '1 / -1';
                note.style.padding = '1rem';
                note.style.background = 'rgba(255,255,255,0.03)';
                note.style.borderRadius = '8px';
                note.style.textAlign = 'center';
                note.style.fontSize = '0.85rem';
                note.style.color = 'var(--text-muted)';
                note.innerHTML = `Showing first 100 image thumbnails of ${resultsList.length.toLocaleString()} total to maintain page speed. The metrics tabs contain full evaluation analytics.`;
                imageGallery.insertBefore(note, imageGallery.firstChild);
            }

            updateMetricsDisplay();

            const activeSubtab = document.querySelector('.subnav-btn.active').getAttribute('data-subtab');
            if (activeSubtab === 'matrix-tab') {
                renderConfusionMatrix();
            } else if (activeSubtab === 'perclass-tab') {
                renderPerClassMetrics();
            } else if (activeSubtab === 'misclassified-tab') {
                renderMisclassifiedList();
            }
            console.log(`Successfully restored ${resultsList.length} evaluation predictions from server cache.`);
        }
    } catch (err) {
        console.error('Failed to load cached evaluation:', err);
    }
}
