// Global state
let cameraActive = false;
let analysisInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    checkCameraStatus();
    setupImageUpload();
    setupAddPersonForm();
    loadSnapshots();
    loadPeople();
});

// fetch() wrapper: redirects to the login page if the session has expired
async function apiFetch(url, options) {
    const response = await fetch(url, options);
    if (response.status === 401) {
        window.location.href = '/login';
        throw new Error('Not authenticated');
    }
    return response;
}

// Start camera
async function startCamera() {
    showLoading(true);
    
    try {
        const response = await apiFetch('/start_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_id: 0 })
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
        
        // Update faces list
        updateFacesList(data.faces);
        
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
        return;
    }
    
    facesList.innerHTML = faces.map(face => `
        <div class="face-item">
            <div class="face-info">
                <div class="face-id">
                    Track ${face.track_id}
                    ${identityBadge(face)}
                </div>
                <div class="face-emotion">
                    ${getEmotionEmoji(face.dominant_emotion)}
                    ${face.dominant_emotion.charAt(0).toUpperCase() + face.dominant_emotion.slice(1)}
                </div>
            </div>
            <div class="face-confidence">${face.confidence.toFixed(1)}%</div>
        </div>
    `).join('');
}

// Render the recognized-name / unknown badge for a tracked face
function identityBadge(face) {
    if (face.name) {
        return `<span class="identity-badge recognized">${face.name}</span>`;
    }
    return `<span class="identity-badge unknown">Unknown</span>`;
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
    
    showLoading(true);
    
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

// Load snapshots
async function loadSnapshots() {
    try {
        const response = await apiFetch('/get_snapshots');
        const data = await response.json();
        
        const grid = document.getElementById('snapshotsGrid');
        
        if (data.success && data.snapshots.length > 0) {
            grid.innerHTML = data.snapshots.map(snapshot => `
                <div class="snapshot-item" onclick="viewSnapshot('${snapshot.path}')">
                    <img src="${snapshot.path}" alt="Snapshot">
                </div>
            `).join('');
        } else {
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

// View snapshot (open in new tab)
function viewSnapshot(path) {
    window.open(path, '_blank');
}

// Wire up the "Add Person" enrollment form
function setupAddPersonForm() {
    const form = document.getElementById('addPersonForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const name = document.getElementById('personName').value.trim();
        const photoFiles = document.getElementById('personPhotos').files;

        if (!name || photoFiles.length === 0) {
            showToast('Enter a name and choose at least one photo', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('name', name);
        for (const file of photoFiles) {
            formData.append('photos', file);
        }

        showLoading(true);
        try {
            const response = await apiFetch('/people', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.success) {
                const failed = data.photos_failed ? data.photos_failed.length : 0;
                const message = failed > 0
                    ? `Enrolled ${name} (${data.photos_enrolled} photo(s), ${failed} had no detectable face)`
                    : `Enrolled ${name} (${data.photos_enrolled} photo(s))`;
                showToast(message, 'success');
                form.reset();
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

// Load enrolled people into the panel
async function loadPeople() {
    try {
        const response = await apiFetch('/people');
        const data = await response.json();

        const grid = document.getElementById('peopleGrid');

        if (data.success && data.people.length > 0) {
            grid.innerHTML = data.people.map(person => `
                <div class="person-card">
                    ${person.thumbnail
                        ? `<img src="/people/${encodeURIComponent(person.name)}/photo/${encodeURIComponent(person.thumbnail)}" alt="${person.name}">`
                        : `<img alt="${person.name}">`}
                    <div class="person-name">${person.name}</div>
                    <div class="person-photos">${person.num_photos} photo(s)</div>
                    <button class="person-delete-btn" onclick="deletePerson('${person.name}')">Remove</button>
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

// Show/hide loading overlay
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

// Handle visibility change (pause updates when tab not visible)
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        stopAnalysisUpdates();
    } else if (cameraActive) {
        startAnalysisUpdates();
    }
});
