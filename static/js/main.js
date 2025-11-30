// Disease Prediction System - Main JavaScript
// Hỗ trợ tiếng Việt

// Global variables
let allSymptoms = [];
let selectedSymptoms = new Set();

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function () {
    console.log('🚀 Disease Prediction System initialized');
    loadSymptoms();
    setupEventListeners();
});

// Load symptoms from backend
async function loadSymptoms() {
    try {
        const response = await fetch('/api/symptoms');
        const data = await response.json();

        if (data.success) {
            allSymptoms = data.symptoms;
            renderSymptoms(allSymptoms);
            console.log(`✅ Loaded ${allSymptoms.length} symptoms`);
        } else {
            showError('Không thể tải danh sách triệu chứng');
        }
    } catch (error) {
        console.error('Error loading symptoms:', error);
        showError('Lỗi kết nối đến server');
    }
}

// Render symptoms to grid
function renderSymptoms(symptoms) {
    const grid = document.getElementById('symptomsGrid');
    grid.innerHTML = '';

    symptoms.forEach(symptom => {
        const div = document.createElement('div');
        div.className = 'symptom-item';
        div.setAttribute('data-symptom', symptom.en);

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `symptom-${symptom.en.replace(/[^a-zA-Z0-9]/g, '-')}`;
        checkbox.onchange = () => toggleSymptom(symptom.en);

        const label = document.createElement('label');
        label.setAttribute('for', checkbox.id);

        // Hiển thị tiếng Việt và tiếng Anh
        label.innerHTML = `
            <div class="symptom-name-vi">${symptom.vi}</div>
            <div class="symptom-name-en">${symptom.en}</div>
        `;

        div.appendChild(checkbox);
        div.appendChild(label);
        grid.appendChild(div);
    });

    updateSearchStats(symptoms.length);
}

// Toggle symptom selection
function toggleSymptom(symptomEn) {
    const item = document.querySelector(`.symptom-item[data-symptom="${symptomEn}"]`);
    const checkbox = item.querySelector('input[type="checkbox"]');

    if (selectedSymptoms.has(symptomEn)) {
        selectedSymptoms.delete(symptomEn);
        item.classList.remove('selected');
    } else {
        selectedSymptoms.add(symptomEn);
        item.classList.add('selected');
    }

    updateSelectedSymptoms();
}

// Update selected symptoms display
function updateSelectedSymptoms() {
    const list = document.getElementById('selectedSymptomsList');
    const count = document.getElementById('selectedCount');
    const predictBtn = document.getElementById('predictBtn');

    count.textContent = selectedSymptoms.size;

    if (selectedSymptoms.size === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">🩺</span>
                <p>Chưa có triệu chứng nào được chọn</p>
                <small>Hãy chọn triệu chứng từ danh sách bên dưới</small>
            </div>
        `;
        predictBtn.disabled = true;
    } else {
        list.innerHTML = '';
        selectedSymptoms.forEach(symptomEn => {
            // Tìm tên tiếng Việt
            const symptom = allSymptoms.find(s => s.en === symptomEn);
            const symptomVi = symptom ? symptom.vi : symptomEn;

            const tag = document.createElement('span');
            tag.className = 'symptom-tag';

            const text = document.createElement('span');
            text.innerHTML = `<strong>${symptomVi}</strong><br><small>${symptomEn}</small>`;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-btn';
            removeBtn.innerHTML = '×';
            removeBtn.onclick = () => removeSymptom(symptomEn);

            tag.appendChild(text);
            tag.appendChild(removeBtn);
            list.appendChild(tag);
        });
        predictBtn.disabled = false;
    }
}

// Remove symptom
function removeSymptom(symptomEn) {
    selectedSymptoms.delete(symptomEn);

    const item = document.querySelector(`.symptom-item[data-symptom="${symptomEn}"]`);
    if (item) {
        const checkbox = item.querySelector('input[type="checkbox"]');
        checkbox.checked = false;
        item.classList.remove('selected');
    }

    updateSelectedSymptoms();
}

// Clear all selected symptoms
function clearAll() {
    selectedSymptoms.clear();

    document.querySelectorAll('.symptom-item').forEach(item => {
        const checkbox = item.querySelector('input[type="checkbox"]');
        checkbox.checked = false;
        item.classList.remove('selected');
    });

    updateSelectedSymptoms();
}

// Setup event listeners
function setupEventListeners() {
    const searchInput = document.getElementById('searchInput');

    searchInput.addEventListener('input', function (e) {
        const searchTerm = e.target.value.toLowerCase().trim();

        if (searchTerm === '') {
            renderSymptoms(allSymptoms);
        } else {
            // Tìm kiếm cả tiếng Việt và tiếng Anh
            const filtered = allSymptoms.filter(symptom =>
                symptom.vi.toLowerCase().includes(searchTerm) ||
                symptom.en.toLowerCase().includes(searchTerm)
            );
            renderSymptoms(filtered);
        }

        // Restore selected state
        selectedSymptoms.forEach(symptomEn => {
            const item = document.querySelector(`.symptom-item[data-symptom="${symptomEn}"]`);
            if (item) {
                item.classList.add('selected');
                item.querySelector('input[type="checkbox"]').checked = true;
            }
        });
    });
}

// Update search stats
function updateSearchStats(count) {
    const stats = document.getElementById('searchResults');
    stats.textContent = `Hiển thị ${count} triệu chứng`;
}

// Predict disease
async function predictDisease() {
    const loading = document.getElementById('loading');
    const results = document.getElementById('resultsSection');
    const errorAlert = document.getElementById('errorAlert');
    const predictBtn = document.getElementById('predictBtn');

    // Hide previous results and errors
    results.classList.remove('show');
    errorAlert.classList.remove('show');

    // Show loading
    loading.classList.add('show');
    predictBtn.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                symptoms: Array.from(selectedSymptoms)
            })
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data.results, data.timestamp);
        } else {
            throw new Error(data.error || 'Có lỗi xảy ra');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message);
    } finally {
        loading.classList.remove('show');
        predictBtn.disabled = false;
    }
}

// Display results
function displayResults(results, timestamp) {
    const resultsSection = document.getElementById('resultsSection');
    const primaryResult = document.getElementById('primaryResult');
    const predictionsList = document.getElementById('predictionsList');
    const symptomsMatched = document.getElementById('symptomsMatched');
    const predictionTime = document.getElementById('predictionTime');

    // Primary result
    const confidence = results.confidence * 100;
    primaryResult.innerHTML = `
        <h3>🎯 Kết quả dự đoán chính</h3>
        <div class="disease-name">${results.primary_prediction_vi}</div>
        <div class="disease-name-en">${results.primary_prediction}</div>
        <div style="font-size: 1.2em; margin-top: 10px;">
            <strong>Độ tin cậy:</strong> ${confidence.toFixed(1)}%
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence}%"></div>
        </div>
    `;

    // Info
    symptomsMatched.textContent = results.matched_symptoms.length;
    predictionTime.textContent = new Date(timestamp).toLocaleString('vi-VN');

    // Top predictions
    predictionsList.innerHTML = '';
    results.top_predictions.forEach((prediction, index) => {
        const card = document.createElement('div');
        card.className = `prediction-card rank-${index + 1}`;

        const percentage = (prediction.probability * 100).toFixed(1);

        card.innerHTML = `
            <div class="prediction-header">
                <span class="prediction-rank">#${index + 1}</span>
                <div class="prediction-name-container">
                    <span class="prediction-name">${prediction.disease_vi}</span>
                    <span class="prediction-name-en">${prediction.disease}</span>
                </div>
                <span class="prediction-percentage">${percentage}%</span>
            </div>
            <div class="prediction-bar">
                <div class="prediction-fill" style="width: ${percentage}%"></div>
            </div>
        `;

        predictionsList.appendChild(card);
    });

    // Show results with animation
    setTimeout(() => {
        resultsSection.classList.add('show');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Close results
function closeResults() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('show');
}

// Show error
function showError(message) {
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');

    errorMessage.textContent = message;
    errorAlert.classList.add('show');

    // Auto hide after 5 seconds
    setTimeout(() => {
        errorAlert.classList.remove('show');
    }, 5000);
}

// Show about dialog
function showAbout() {
    alert(`
🏥 Hệ Thống Dự Đoán Bệnh
        
Phiên bản: 1.0.0
Công nghệ: AI & Machine Learning

Được phát triển bởi sinh viên IT
Sử dụng Random Forest/XGBoost Classifier

⚠️ Lưu ý: Đây chỉ là công cụ hỗ trợ tham khảo.
Vui lòng đến cơ sở y tế để được chẩn đoán chính xác!
    `);
}

// Export functions for inline event handlers
window.predictDisease = predictDisease;
window.clearAll = clearAll;
window.closeResults = closeResults;
window.showAbout = showAbout;