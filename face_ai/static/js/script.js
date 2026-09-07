// Global state
let cameraActive = false;
let analysisInterval = null;

// Tracks each face's previous "resolved" state (recognized_once), keyed by
// track_id, so we can detect the exact poll where a scan finishes and add a
// one-shot flash animation instead of re-flashing on every 500ms refresh.
let previousResolvedState = new Map();

// Rolling emotion-distribution history for the sparkline (~20s at 500ms polling)
const EMOTION_COLORS = {
    happy: '#22c55e', sad: '#3b82f6', angry: '#ef4444', surprise: '#eab308',
    fear: '#a855f7', disgust: '#065f46', neutral: '#9ca3af'
};
const MAX_SPARKLINE_SAMPLES = 40;
let emotionHistory = [];

// Names currently enrolled, used to power the Add Person autocomplete and
// to detect "adding photos to an existing person" vs. a brand-new one.
let enrolledNames = [];

function recordEmotionSample(distribution) {
    emotionHistory.push(distribution);
    if (emotionHistory.length > MAX_SPARKLINE_SAMPLES) emotionHistory.shift();
    drawEmotionSparkline();
}

function drawEmotionSparkline() {
    const canvas = document.getElementById('emotionSparkline');
    if (!canvas) return;

    const width = canvas.clientWidth;
    const height = canvas.clientHeight || 60;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    if (emotionHistory.length < 2) return;

    const stepX = width / (MAX_SPARKLINE_SAMPLES - 1);
    const offset = MAX_SPARKLINE_SAMPLES - emotionHistory.length; // right-align newest data

    Object.entries(EMOTION_COLORS).forEach(([emotion, color]) => {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        emotionHistory.forEach((sample, i) => {
            const x = (offset + i) * stepX;
            const pct = (sample[emotion] || 0) / 100;
            const y = height - (pct * (height - 4)) - 2;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkCameraStatus();
    setupImageUpload();
    setupAddPersonForm();
    setupThresholdSlider();
    loadCameraOptions();
    loadSnapshots();
    loadPeople();
});

// Populate the camera picker. Skipped (left as-is) while a camera is
// already running, since probing would fight it for the device.
async function loadCameraOptions() {
    const select = document.getElementById('cameraSelect');
    try {
        const response = await apiFetch('/cameras');
        const data = await response.json();
        if (!data.success || data.busy) return;

        if (data.cameras.length === 0) {
            select.innerHTML = '<option value="0">No camera found</option>';
            return;
        }
        select.innerHTML = data.cameras.map(id => `<option value="${id}">Camera ${id}</option>`).join('');
    } catch (error) {
        console.error('Load cameras error:', error);
    }
}

// Load the current match threshold and wire the slider to /settings
async function setupThresholdSlider() {
    const slider = document.getElementById('thresholdSlider');
    const valueLabel = document.getElementById('thresholdValue');

    try {
        const response = await apiFetch('/settings');
        const data = await response.json();
        if (data.success) {
            slider.value = data.match_threshold;
            valueLabel.textContent = data.match_threshold.toFixed(3);
        }
    } catch (error) {
        console.error('Load settings error:', error);
    }

    // Update the label continuously while dragging, but only push the
    // change to the server once the user releases the slider.
    slider.addEventListener('input', () => {
        valueLabel.textContent = parseFloat(slider.value).toFixed(3);
    });

    slider.addEventListener('change', async () => {
        try {
            const response = await apiFetch('/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ match_threshold: parseFloat(slider.value) })
            });
            const data = await response.json();
            if (data.success) {
                showToast(`Match threshold set to ${data.match_threshold.toFixed(3)}`, 'success');
            } else {
                throw new Error(data.error || 'Failed to update threshold');
            }
        } catch (error) {
            showToast(`Error: ${error.message}`, 'error');
        }
    });
}

// fetch() wrapper: attaches the CSRF token to unsafe requests and redirects
// to the login page if the session has expired
async function apiFetch(url, options = {}) {
    const method = (options.method || 'GET').toUpperCase();

    if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
        const meta = document.querySelector('meta[name="csrf-token"]');
        const token = meta ? meta.content : '';
        options = {
            ...options,
            headers: {
                ...(options.headers || {}),
                'X-CSRFToken': token
            }
        };
    }

    const response = await fetch(url, options);
    if (response.status === 401) {
        window.location.href = '/login';
        throw new Error('Not authenticated');
    }
    return response;
}

// Start camera
async function startCamera() {
    showLoading(true, 'Starting camera...');

    try {
        const cameraId = parseInt(document.getElementById('cameraSelect').value || '0', 10);
        const response = await apiFetch('/start_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_id: cameraId })
        });

        const data = await response.json();
        
        if (data.success) {
            cameraActive = true;
            updateCameraUI(true);
            startVideoFeed();
            startAnalysisUpdates();
            showToast('Camera started successfully!', 'success');
        } else {
            throw new Error(data.error || 'Failed to start camera');
        }
        
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
        console.error('Start camera error:', error);
    } finally {
        showLoading(false);
    }
}

// Stop camera
async function stopCamera() {
    try {
        const response = await apiFetch('/stop_camera', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            cameraActive = false;
            updateCameraUI(false);
            stopVideoFeed();
            stopAnalysisUpdates();
            resetAnalysis();
            showToast('Camera stopped', 'success');
        }
        
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

// Start video feed
function startVideoFeed() {
    const videoFeed = document.getElementById('videoFeed');
    const overlay = document.getElementById('noCamera');
    
    videoFeed.src = '/video_feed?' + new Date().getTime();
    overlay.style.display = 'none';
}

// Stop video feed
function stopVideoFeed() {
    const videoFeed = document.getElementById('videoFeed');
    const overlay = document.getElementById('noCamera');
    
    videoFeed.src = '/static/images/placeholder.jpg';
    overlay.style.display = 'flex';
}

// Update camera UI
function updateCameraUI(active) {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const captureBtn = document.getElementById('captureBtn');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    if (active) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        captureBtn.disabled = false;
        statusDot.className = 'status-dot online';
        statusText.textContent = 'Camera Active';
    } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        captureBtn.disabled = true;
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'Camera Offline';
    }
}

// Start analysis updates
function startAnalysisUpdates() {
    analysisInterval = setInterval(updateAnalysis, 500); // Update every 500ms
}

// Stop analysis updates
function stopAnalysisUpdates() {
    if (analysisInterval) {
        clearInterval(analysisInterval);
        analysisInterval = null;
    }
}

// Update analysis data
async function updateAnalysis() {
    if (!cameraActive) return;
    
    try {
        const response = await apiFetch('/get_analysis');
        const data = await response.json();
        
        // Update face count
        document.getElementById('faceCount').textContent = data.num_faces;
        
        // Update dominant emotion
        if (data.num_faces > 0 && data.faces.length > 0) {
            const dominantEmotion = data.faces[0].dominant_emotion;
            document.getElementById('dominantEmotion').textContent = 
                dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1);
        } else {
            document.getElementById('dominantEmotion').textContent = '-';
        }
        
        // Update emotion bars
        updateEmotionBars(data.emotion_distribution);
        recordEmotionSample(data.emotion_distribution);

        // Update faces list
        updateFacesList(data.faces);

        // Update FPS / latency and liveness summary
        document.getElementById('fpsValue').textContent =
            data.num_faces > 0 ? `${data.fps} / ${data.processing_ms}ms` : '-';

        const liveness = data.liveness_summary || { live: 0, low_motion: 0, checking: 0 };
        document.getElementById('livenessSummary').textContent =
            data.num_faces > 0 ? `${liveness.live} / ${liveness.low_motion}` : '-';

    } catch (error) {
        console.error('Update analysis error:', error);
    }
}

// Update emotion bars
function updateEmotionBars(emotions) {
    for (const [emotion, value] of Object.entries(emotions)) {
        const item = document.querySelector(`.emotion-item[data-emotion="${emotion}"]`);
        if (item) {
            const bar = item.querySelector('.emotion-bar');
            const valueSpan = item.querySelector('.emotion-value');
            
            const percentage = Math.round(value);
            bar.style.width = percentage + '%';
            valueSpan.textContent = percentage + '%';
        }
    }
}

// Update faces list
function updateFacesList(faces) {
    const facesList = document.getElementById('facesList');

    if (faces.length === 0) {
        facesList.innerHTML = `
            <div class="empty-state">
                <p>No faces detected</p>
                <p class="empty-hint">Position face in camera view</p>
            </div>
        `;
        previousResolvedState.clear();
        return;
    }

    facesList.innerHTML = faces.map(face => {
        // A track "just resolved" the exact poll its first recognition pass
        // completes — flash it once so the scan-to-result moment is visible,
        // instead of just silently swapping the badge on the next refresh.
        const wasResolved = previousResolvedState.get(face.track_id);
        const justResolved = face.recognized_once && wasResolved === false;
        previousResolvedState.set(face.track_id, !!face.recognized_once);

        const stateClass = !face.recognized_once ? 'scanning' : (justResolved ? 'just-matched' : '');

        return `
        <div class="face-item ${stateClass}">
            <div class="face-info">
                <div class="face-id">
                    Track ${face.track_id}
                    ${identityBadge(face)}
                    ${livenessBadge(face)}
                </div>
                <div class="face-emotion">
                    ${getEmotionEmoji(face.dominant_emotion)}
                    ${face.dominant_emotion.charAt(0).toUpperCase() + face.dominant_emotion.slice(1)}
                </div>
            </div>
            <div class="face-confidence">${face.confidence.toFixed(1)}%</div>
        </div>
    `;
    }).join('');

    // Drop bookkeeping for tracks that left the frame so the map doesn't
    // grow for the lifetime of a long camera session.
    const liveTrackIds = new Set(faces.map(f => f.track_id));
    for (const trackId of previousResolvedState.keys()) {
        if (!liveTrackIds.has(trackId)) previousResolvedState.delete(trackId);
    }
}

// Render the recognized-name / unknown / still-scanning badge for a tracked face
function identityBadge(face) {
    if (!face.recognized_once) {
        return `<span class="identity-badge scanning">🔍 Scanning...</span>`;
    }
    if (face.name) {
        return `<span class="identity-badge recognized">✓ ${face.name}</span>`;
    }
    return `<span class="identity-badge unknown">Unknown</span>`;
}

// Surface what the liveness heuristic is doing: progress while it's still
// collecting motion samples, a warning if the face looks suspiciously
// static once it has enough samples (e.g. a printed photo held to the
// camera). Quiet (no badge) once confirmed live — nothing to flag.
function livenessBadge(face) {
    if (face.liveness_status === 'low_motion') {
        return `<span class="identity-badge spoof-warning" title="No motion detected across recent frames">⚠ possible spoof</span>`;
    }
    if (face.liveness_status === 'checking' && face.recognized_once) {
        const pct = Math.round((face.liveness_progress || 0) * 100);
        return `<span class="identity-badge checking" title="Confirming this is a live face, not a photo">verifying ${pct}%</span>`;
    }
    return '';
}

// Get emotion emoji
function getEmotionEmoji(emotion) {
    const emojis = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'surprise': '😲',
        'fear': '😨',
        'disgust': '🤢',
        'neutral': '😐'
    };
    return emojis[emotion] || '😐';
}

// Reset analysis display
function resetAnalysis() {
    document.getElementById('faceCount').textContent = '0';
    document.getElementById('dominantEmotion').textContent = '-';
    document.getElementById('fpsValue').textContent = '-';
    document.getElementById('livenessSummary').textContent = '-';

    // Reset emotion bars
    document.querySelectorAll('.emotion-bar').forEach(bar => {
        bar.style.width = '0%';
    });
    document.querySelectorAll('.emotion-value').forEach(value => {
        value.textContent = '0%';
    });

    // Reset faces list
    document.getElementById('facesList').innerHTML = `
        <div class="empty-state">
            <p>No faces detected yet</p>
            <p class="empty-hint">Start camera or upload an image</p>
        </div>
    `;
    previousResolvedState.clear();

    emotionHistory = [];
    drawEmotionSparkline();
}

// Capture snapshot
async function captureSnapshot() {
    if (!cameraActive) {
        showToast('Camera is not active', 'error');
        return;
    }
    
    try {
        const response = await apiFetch('/capture_snapshot', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Snapshot captured!', 'success');
            loadSnapshots();
        } else {
            throw new Error(data.error || 'Failed to capture');
        }
        
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

// Setup image upload
function setupImageUpload() {
    const imageInput = document.getElementById('imageInput');
    const uploadArea = document.getElementById('uploadArea');
    
    imageInput.addEventListener('change', handleImageUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#cbd5e0';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#cbd5e0';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            handleImageUpload({ target: imageInput });
        }
    });
}

// Handle image upload
async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    showLoading(true, 'Scanning image for faces...');

    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await apiFetch('/analyze_image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Display result
            const resultDiv = document.getElementById('uploadResult');
            resultDiv.innerHTML = `
                <img src="${data.result_path}" alt="Analysis Result">
                <div style="margin-top: 15px; padding: 15px; background: #f7fafc; border-radius: 10px;">
                    <strong>Detected ${data.analysis.num_faces} face(s)</strong>
                    ${data.analysis.faces.map(face => `
                        <div style="margin-top: 10px;">
                            Face ${face.face_id}: ${face.name ? face.name : 'Unknown'} &mdash; ${face.emotion} (${face.confidence.toFixed(1)}%)
                        </div>
                    `).join('')}
                </div>
            `;
            resultDiv.style.display = 'block';
            
            showToast('Image analyzed successfully!', 'success');
            loadSnapshots();
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
        
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
        event.target.value = '';
    }
}

// The currently loaded snapshot list, kept around so the lightbox can
// navigate prev/next without a round-trip per click.
let currentSnapshots = [];
let lightboxIndex = -1;

// Load snapshots
async function loadSnapshots() {
    try {
        const response = await apiFetch('/get_snapshots');
        const data = await response.json();

        const grid = document.getElementById('snapshotsGrid');

        if (data.success && data.snapshots.length > 0) {
            currentSnapshots = data.snapshots;
            grid.innerHTML = data.snapshots.map((snapshot, i) => `
                <div class="snapshot-item">
                    <img src="${snapshot.path}" alt="Snapshot" onclick="openLightbox(${i})">
                    <button class="snapshot-delete-btn" title="Delete"
                            onclick="event.stopPropagation(); deleteSnapshot('${snapshot.filename}')">&times;</button>
                </div>
            `).join('');
        } else {
            currentSnapshots = [];
            grid.innerHTML = `
                <div class="empty-state">
                    <p>No captures yet</p>
                    <p class="empty-hint">Use the capture button to save moments</p>
                </div>
            `;
        }

    } catch (error) {
        console.error('Load snapshots error:', error);
    }
}

async function deleteSnapshot(filename) {
    try {
        const response = await apiFetch(`/media/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        const data = await response.json();
        if (data.success) {
            loadSnapshots();
        } else {
            throw new Error(data.error || 'Failed to delete snapshot');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

function openLightbox(index) {
    lightboxIndex = index;
    document.getElementById('lightboxImage').src = currentSnapshots[index].path;
    document.getElementById('lightboxModal').style.display = 'flex';
}

function closeLightbox() {
    document.getElementById('lightboxModal').style.display = 'none';
    lightboxIndex = -1;
}

function navigateLightbox(delta) {
    if (lightboxIndex === -1 || currentSnapshots.length === 0) return;
    lightboxIndex = (lightboxIndex + delta + currentSnapshots.length) % currentSnapshots.length;
    document.getElementById('lightboxImage').src = currentSnapshots[lightboxIndex].path;
}

// Keyboard navigation for the lightbox (arrows to move, Escape to close)
document.addEventListener('keydown', (e) => {
    if (lightboxIndex === -1) return;
    if (e.key === 'ArrowLeft') navigateLightbox(-1);
    else if (e.key === 'ArrowRight') navigateLightbox(1);
    else if (e.key === 'Escape') closeLightbox();
});

// Wire up the "Add Person" enrollment form
function setupAddPersonForm() {
    const form = document.getElementById('addPersonForm');
    const nameInput = document.getElementById('personName');
    const submitBtn = document.getElementById('addPersonSubmitBtn');

    // Switch the button label to make it clear that typing an existing
    // name adds photos to that person instead of creating a duplicate.
    nameInput.addEventListener('input', () => {
        const isExisting = enrolledNames.includes(nameInput.value.trim());
        submitBtn.innerHTML = isExisting
            ? '<span class="btn-icon">📷</span> Add Photos'
            : '<span class="btn-icon">➕</span> Add Person';
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const name = nameInput.value.trim();
        const photoFiles = document.getElementById('personPhotos').files;
        const isExisting = enrolledNames.includes(name);

        if (!name || photoFiles.length === 0) {
            showToast('Enter a name and choose at least one photo', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('name', name);
        for (const file of photoFiles) {
            formData.append('photos', file);
        }

        showLoading(true, 'Scanning photos and enrolling...');
        try {
            const response = await apiFetch('/people', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.success) {
                const failed = data.photos_failed ? data.photos_failed.length : 0;
                const verb = isExisting ? 'Added to' : 'Enrolled';
                const message = failed > 0
                    ? `${verb} ${name} (${data.photos_enrolled} photo(s), ${failed} had no detectable face)`
                    : `${verb} ${name} (${data.photos_enrolled} photo(s))`;
                showToast(message, 'success');
                form.reset();
                submitBtn.innerHTML = '<span class="btn-icon">➕</span> Add Person';
                loadPeople();
            } else {
                throw new Error(data.error || 'Failed to enroll person');
            }
        } catch (error) {
            showToast(`Error: ${error.message}`, 'error');
        } finally {
            showLoading(false);
        }
    });
}

// Human-readable "time since" for last-seen timestamps (epoch seconds).
function formatTimeAgo(epochSeconds) {
    if (!epochSeconds) return 'Never seen';
    const seconds = Math.max(0, (Date.now() / 1000) - epochSeconds);
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

// Load enrolled people into the panel
async function loadPeople() {
    try {
        const response = await apiFetch('/people');
        const data = await response.json();

        const grid = document.getElementById('peopleGrid');

        if (data.success) {
            enrolledNames = data.people.map(p => p.name);
            document.getElementById('enrolledNamesList').innerHTML =
                enrolledNames.map(name => `<option value="${name}">`).join('');
        }

        if (data.success && data.people.length > 0) {
            grid.innerHTML = data.people.map(person => `
                <div class="person-card">
                    ${person.thumbnail
                        ? `<img src="/people/${encodeURIComponent(person.name)}/photo/${encodeURIComponent(person.thumbnail)}" alt="${person.name}">`
                        : `<img alt="${person.name}">`}
                    <div class="person-name">${person.name}</div>
                    <div class="person-photos">${person.num_photos} photo(s)</div>
                    <div class="person-last-seen">${formatTimeAgo(person.last_seen)}</div>
                    <div class="person-card-actions">
                        <button class="person-manage-btn" onclick="openPhotoManager('${person.name}')">Photos</button>
                        <button class="person-delete-btn" onclick="deletePerson('${person.name}')">Remove</button>
                    </div>
                </div>
            `).join('');
        } else {
            grid.innerHTML = `
                <div class="empty-state">
                    <p>No one enrolled yet</p>
                    <p class="empty-hint">Add a person below to start recognizing them</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Load people error:', error);
    }
}

// Remove an enrolled person
async function deletePerson(name) {
    try {
        const response = await apiFetch(`/people/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const data = await response.json();

        if (data.success) {
            showToast(`Removed ${name}`, 'success');
            loadPeople();
        } else {
            throw new Error(data.error || 'Failed to remove person');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

// Which person's photo-manager modal is currently open, so a delete inside
// it knows whose photo list to re-fetch and re-render.
let managingPersonName = null;

async function openPhotoManager(name) {
    managingPersonName = name;
    document.getElementById('photoManagerTitle').textContent = `${name}'s photos`;
    document.getElementById('photoManagerModal').style.display = 'flex';
    await renderPhotoManager();
}

function closePhotoManager() {
    managingPersonName = null;
    document.getElementById('photoManagerModal').style.display = 'none';
}

async function renderPhotoManager() {
    const name = managingPersonName;
    if (!name) return;

    const grid = document.getElementById('photoManagerGrid');
    grid.innerHTML = '<p class="empty-hint">Loading...</p>';

    try {
        const response = await apiFetch(`/people/${encodeURIComponent(name)}/photos`);
        const data = await response.json();

        if (!data.success || data.photos.length === 0) {
            grid.innerHTML = '<p class="empty-hint">No photos left</p>';
            return;
        }

        grid.innerHTML = data.photos.map(filename => `
            <div class="photo-manager-item">
                <img src="/people/${encodeURIComponent(name)}/photo/${encodeURIComponent(filename)}" alt="${name}">
                <button class="photo-delete-btn" title="Remove this photo"
                        onclick="deletePersonPhoto('${filename}')">&times;</button>
            </div>
        `).join('');
    } catch (error) {
        grid.innerHTML = '<p class="empty-hint">Failed to load photos</p>';
    }
}

async function deletePersonPhoto(filename) {
    const name = managingPersonName;
    if (!name) return;

    try {
        const response = await apiFetch(
            `/people/${encodeURIComponent(name)}/photos/${encodeURIComponent(filename)}`,
            { method: 'DELETE' }
        );
        const data = await response.json();

        if (data.success) {
            loadPeople();
            if (data.person_removed) {
                showToast(`${name} removed (no photos left)`, 'success');
                closePhotoManager();
            } else {
                await renderPhotoManager();
            }
        } else {
            throw new Error(data.error || 'Failed to remove photo');
        }
    } catch (error) {
        showToast(`Error: ${error.message}`, 'error');
    }
}

// Check camera status
async function checkCameraStatus() {
    try {
        const response = await apiFetch('/camera_status');
        const data = await response.json();
        
        if (data.is_running) {
            cameraActive = true;
            updateCameraUI(true);
            startVideoFeed();
            startAnalysisUpdates();
        }
    } catch (error) {
        console.error('Check status error:', error);
    }
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Show/hide the scanning-themed loading overlay, with an optional message
// tailored to what's actually happening (starting the camera, scanning an
// uploaded image, etc.) instead of a generic "Processing...".
function showLoading(show, message = 'Processing...') {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
    document.getElementById('loadingMessage').textContent = message;
}

// Handle visibility change (pause updates when tab not visible)
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopAnalysisUpdates();
    } else if (cameraActive) {
        startAnalysisUpdates();
    }
});
