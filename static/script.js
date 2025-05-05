document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const dropArea = document.getElementById('dropArea');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const removeFile = document.getElementById('removeFile');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsContent = document.getElementById('resultsContent');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('border-blue-500', 'bg-blue-50');
    }

    function unhighlight() {
        dropArea.classList.remove('border-blue-500', 'bg-blue-50');
    }

    // Handle file drop
    dropArea.addEventListener('drop', handleDrop, false);
    dropArea.addEventListener('click', () => fileInput.click());

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFiles(files);
        }
    }

    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFiles(this.files);
        }
    });

    // Remove file selection
    removeFile.addEventListener('click', function() {
        resetFileInput();
    });

    function resetFileInput() {
        fileInput.value = '';
        fileInfo.classList.add('hidden');
        submitBtn.disabled = true;
        resultsContainer.classList.add('hidden');
    }

    // Handle selected files
    function handleFiles(files) {
        const file = files[0];
        const fileSize = file.size / 1024 / 1024; // in MB
        
        if (fileSize > 50) {
            showError('File size exceeds 50MB limit');
            return;
        }

        fileName.textContent = file.name;
        fileInfo.classList.remove('hidden');
        submitBtn.disabled = false;
    }

    // Form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files.length) return;
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Show loading state
        loadingIndicator.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        
        // Send to server
        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            showError(error.message || 'An error occurred during analysis');
        })
        .finally(() => {
            loadingIndicator.classList.add('hidden');
        });
    });

    // Display results
    function displayResults(data) {
        resultsContent.innerHTML = '';
        
        if (data.error) {
            showError(data.error);
            return;
        }

        resultsContainer.classList.remove('hidden');
        
        // Create result elements based on response
        const resultDiv = document.createElement('div');
        resultDiv.className = 'space-y-3';
        
        // Result status
        const statusDiv = document.createElement('div');
        statusDiv.className = `p-3 rounded-lg ${data.is_deepfake ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`;
        
        const statusIcon = document.createElement('i');
        statusIcon.className = data.is_deepfake ? 'fas fa-exclamation-triangle mr-2' : 'fas fa-check-circle mr-2';
        
        const statusText = document.createElement('span');
        statusText.className = 'font-semibold';
        statusText.textContent = data.message;
        
        statusDiv.appendChild(statusIcon);
        statusDiv.appendChild(statusText);
        resultDiv.appendChild(statusDiv);
        
        // Confidence meter
        const confidenceDiv = document.createElement('div');
        confidenceDiv.className = 'space-y-1';
        
        const confidenceLabel = document.createElement('div');
        confidenceLabel.className = 'flex justify-between text-sm text-gray-600';
        confidenceLabel.innerHTML = '<span>Confidence Level</span><span>' + (data.confidence * 100).toFixed(1) + '%</span>';
        
        const meterDiv = document.createElement('div');
        meterDiv.className = 'w-full bg-gray-200 rounded-full h-2.5';
        
        const meterBar = document.createElement('div');
        meterBar.className = 'bg-blue-600 h-2.5 rounded-full';
        meterBar.style.width = (data.confidence * 100) + '%';
        
        meterDiv.appendChild(meterBar);
        confidenceDiv.appendChild(confidenceLabel);
        confidenceDiv.appendChild(meterDiv);
        resultDiv.appendChild(confidenceDiv);
        
        // Image result if available
        if (data.result_path) {
            const imgDiv = document.createElement('div');
            imgDiv.className = 'mt-4 border rounded-lg overflow-hidden';
            
            const img = document.createElement('img');
            img.src = data.result_path;
            img.alt = 'Analysis Result';
            img.className = 'w-full';
            
            imgDiv.appendChild(img);
            resultDiv.appendChild(imgDiv);
        }
        
        resultsContent.appendChild(resultDiv);
    }

    // Show error message
    function showError(message) {
        resultsContainer.classList.remove('hidden');
        resultsContent.innerHTML = `
            <div class="p-3 bg-red-100 text-red-800 rounded-lg">
                <i class="fas fa-exclamation-circle mr-2"></i>
                <span class="font-semibold">${message}</span>
            </div>
        `;
    }
});