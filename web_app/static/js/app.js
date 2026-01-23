// app.js - Aplikasi prediksi diabetes berbasis web
document.addEventListener('DOMContentLoaded', function() {
    // Ambil elemen-elemen formulir
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');
    
    // Ambil elemen-elemen hasil
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContainer = document.getElementById('resultContainer');
    const riskStatus = document.getElementById('riskStatus');
    const riskProbability = document.getElementById('riskProbability');
    const clinicalDescription = document.getElementById('clinicalDescription');
    const factorList = document.getElementById('factorList');
    
    // Ambil elemen-elemen informasi model
    const bestModelElement = document.getElementById('bestModel');
    const bestModelSideElement = document.getElementById('bestModelSide');
    const featureSetElement = document.getElementById('featureSet');
    const methodUsedElement = document.getElementById('methodUsed');
    const accuracyElement = document.getElementById('accuracy');
    const f1ScoreElementMain = document.getElementById('f1score');
    const precisionElement = document.getElementById('precision');
    const recallElement = document.getElementById('recall');

    // Ambil elemen-elemen perbandingan performa
    const rfeAccuracyElement = document.getElementById('rfeAccuracy');
    const rfeF1ScoreElement = document.getElementById('rfeF1Score');
    const allFeaturesAccuracyElement = document.getElementById('allFeaturesAccuracy');
    const allFeaturesF1ScoreElement = document.getElementById('allFeaturesF1Score');
    
    // Ambil elemen untuk hasil diagnosa
    const diagnosisResultElement = document.getElementById('diagnosisResult');
    
    // Inisialisasi chart SHAP
    let shapChart = null;
    
    // Muat informasi model terbaik saat halaman dimuat
    loadBestModelInfo();
    
    // Fungsi untuk menangani perubahan jenis kelamin
    function handleGenderChange() {
        const genderSelect = document.getElementById('gender');
        const pregnanciesGroup = document.getElementById('pregnanciesGroup');
        
        if (genderSelect && pregnanciesGroup) {
            if (genderSelect.value === 'male') {
                pregnanciesGroup.style.display = 'none';
                // Set nilai kehamilan ke 0 ketika jenis kelamin laki-laki dipilih
                const pregnanciesInput = document.getElementById('pregnancies');
                if (pregnanciesInput) pregnanciesInput.value = '0';
            } else {
                pregnanciesGroup.style.display = 'block';
            }
        }
    }
    
    // Tambahkan event listener untuk perubahan jenis kelamin
    const genderSelect = document.getElementById('gender');
    if (genderSelect) {
        genderSelect.addEventListener('change', handleGenderChange);
        // Panggil fungsi sekali untuk mengatur tampilan awal
        handleGenderChange();
    }
    
    // Event listener untuk formulir prediksi
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            console.log("Form submitted"); // Debug log
            
            // Tampilkan spinner dan nonaktifkan tombol
            if (btnText) btnText.classList.add('d-none');
            if (spinner) spinner.classList.remove('d-none');
            if (predictBtn) predictBtn.disabled = true;
            
            try {
                // Ambil nilai-nilai input
                const gender = document.getElementById('gender').value;
                const pregnancies = gender === 'male' ? 0 : (parseFloat(document.getElementById('pregnancies').value) || 0);
                
                const inputData = {
                    pregnancies: pregnancies,
                    glucose: parseFloat(document.getElementById('glucose').value) || 0,
                    bloodPressure: parseFloat(document.getElementById('bloodPressure').value) || 0,
                    skinThickness: parseFloat(document.getElementById('skinThickness').value) || 0,
                    insulin: parseFloat(document.getElementById('insulin').value) || 0,
                    bmi: parseFloat(document.getElementById('bmi').value) || 0,
                    dpf: parseFloat(document.getElementById('dpf').value) || 0,
                    age: parseFloat(document.getElementById('age').value) || 0
                };
                
                console.log("Input data:", inputData); // Debug log
                
                // Panggil fungsi prediksi
                const result = await predictDiabetes(inputData);
                console.log("Prediction result:", result); // Debug log
                
                // Tampilkan hasil
                displayResults(result);
            } catch (error) {
                console.error('Error during prediction:', error);
                alert('Terjadi kesalahan saat melakukan prediksi. Silakan coba lagi.' + error.message);
            } finally {
                // Sembunyikan spinner dan aktifkan kembali tombol
                if (btnText) btnText.classList.remove('d-none');
                if (spinner) spinner.classList.add('d-none');
                if (predictBtn) predictBtn.disabled = false;
            }
        });
    }
    
    // Fungsi untuk melakukan prediksi melalui API
    async function predictDiabetes(inputData) {
        try {
            console.log("Calling API with data:", inputData); // Debug log
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputData)
            });
            
            console.log("Response status:", response.status); // Debug log
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('API error response:', errorText);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            
            const result = await response.json();
            console.log("API response:", result); // Debug log
            
            // Return the result with actual SHAP values from the server
            return {
                prediction: result.prediction,
                probability: result.probability,
                shapValues: result.shap_values,
                featureNames: result.feature_names
            };
        } catch (error) {
            console.error('Error calling prediction API:', error);
            throw error;
        }
    }
    
    // Fungsi untuk menampilkan hasil
    async function displayResults(result) {
        console.log("Displaying results:", result); // Debug log
        
        // Sembunyikan placeholder dan tampilkan hasil
        if (resultPlaceholder) resultPlaceholder.classList.add('d-none');
        if (resultContainer) resultContainer.classList.remove('d-none');

        // Tampilkan status risiko
        if (riskStatus) {
            if (result.prediction === 1) {
                riskStatus.textContent = 'Pasien Berisiko Tinggi';
                riskStatus.className = 'h5 mb-0 font-weight-bold text-danger';
            } else {
                riskStatus.textContent = 'Pasien Berisiko Rendah';
                riskStatus.className = 'h5 mb-0 font-weight-bold text-success';
            }
        }

        // Tampilkan probabilitas
        if (riskProbability) {
            riskProbability.textContent = `${(result.probability * 100).toFixed(2)}%`;
        }

        // Tampilkan informasi klinis
        updateClinicalInfo(result);

        // Tampilkan hasil diagnosa
        displayDiagnosisResult(result);

        // Tampilkan faktor-faktor penting
        displayFeatureImportance(result);

        // Gambar grafik SHAP
        drawShapChart(result);
    }


    // Fungsi untuk menampilkan hasil diagnosa
    function displayDiagnosisResult(result) {
        let diagnosisText = '';

        if (result.prediction === 1) {
            diagnosisText = `Berdasarkan analisis terhadap parameter-parameter medis yang dimasukkan, model prediktif mendeteksi adanya indikasi risiko tinggi terkena diabetes. Probabilitas terjadinya diabetes adalah ${(result.probability * 100).toFixed(2)}%.`;
        } else {
            diagnosisText = `Berdasarkan analisis terhadap parameter-parameter medis yang dimasukkan, model prediktif menunjukkan risiko rendah terkena diabetes. Probabilitas terjadinya diabetes adalah ${(result.probability * 100).toFixed(2)}%.`;
        }

        // Tambahkan informasi tambahan berdasarkan faktor-faktor penting
        const featureImpacts = result.featureNames.map((name, index) => ({
            name: name,
            impact: result.shapValues[index]
        })).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

        const topFactor = featureImpacts[0];
        if (topFactor) {
            diagnosisText += ` Faktor paling signifikan yang mempengaruhi prediksi adalah ${topFactor.name} `;
            if (topFactor.impact > 0) {
                diagnosisText += `(berkontribusi meningkatkan risiko).`;
            } else {
                diagnosisText += `(berkontribusi menurunkan risiko).`;
            }
        }

        if (diagnosisResultElement) {
            diagnosisResultElement.textContent = diagnosisText;
        }
    }
    
    // Fungsi untuk memperbarui informasi klinis
    function updateClinicalInfo(result) {
        let infoText = '';

        if (result.prediction === 1) {
            infoText = 'Pasien menunjukkan beberapa faktor risiko yang meningkatkan kemungkinan terkena diabetes. Disarankan untuk melakukan pemeriksaan lebih lanjut dan konsultasi dengan tenaga medis.';
        } else {
            infoText = 'Pasien memiliki faktor risiko yang relatif rendah untuk terkena diabetes. Tetap jaga pola hidup sehat untuk mencegah risiko di masa depan.';
        }

        if (clinicalDescription) {
            clinicalDescription.textContent = infoText;
        }
    }
    
    // Fungsi untuk menampilkan faktor-faktor penting
    function displayFeatureImportance(result) {
        // Kosongkan daftar sebelumnya
        if (factorList) {
            factorList.innerHTML = '';

            // Buat array pasangan [nama fitur, nilai SHAP] dan urutkan berdasarkan nilai absolut
            const featureImpacts = result.featureNames.map((name, index) => ({
                name: name,
                impact: result.shapValues[index]
            })).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

            // Ambil 3 faktor teratas
            const topFactors = featureImpacts.slice(0, 3);

            // Tambahkan setiap faktor ke daftar
            topFactors.forEach(factor => {
                const listItem = document.createElement('li');
                listItem.className = `list-group-item d-flex justify-content-between align-items-center ${factor.impact > 0 ? 'increasing' : 'decreasing'}`;

                const factorName = document.createElement('span');
                factorName.textContent = factor.name;

                const impactValue = document.createElement('span');
                impactValue.className = 'badge bg-secondary rounded-pill';
                impactValue.textContent = factor.impact.toFixed(3);

                listItem.appendChild(factorName);
                listItem.appendChild(impactValue);
                factorList.appendChild(listItem);
            });
        }
    }

    // Fungsi untuk menggambar grafik SHAP
    function drawShapChart(result) {
        const canvasElement = document.getElementById('shapChart');
        if (!canvasElement) {
            console.error('SHAP chart canvas element not found');
            return;
        }
        
        const ctx = canvasElement.getContext('2d');

        // Hapus chart sebelumnya jika ada
        if (shapChart) {
            shapChart.destroy();
        }

        // Siapkan data untuk chart
        const labels = result.featureNames;
        const data = result.shapValues;

        // Warna berdasarkan nilai positif/negatif
        const backgroundColors = data.map(value =>
            value > 0 ? 'rgba(231, 74, 59, 0.6)' : 'rgba(28, 200, 138, 0.6)'
        );

        const borderColors = data.map(value =>
            value > 0 ? 'rgba(231, 74, 59, 1)' : 'rgba(28, 200, 138, 1)'
        );

        // Buat chart baru
        shapChart = new Chart(ctx, {
            type: 'bar', // Changed to 'bar' for better horizontal visualization
            data: {
                labels: labels,
                datasets: [{
                    label: 'Nilai SHAP',
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y', // This makes it a horizontal bar chart
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Nilai SHAP'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `SHAP: ${context.parsed.x.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Fungsi untuk memuat informasi model terbaik
    async function loadBestModelInfo() {
        try {
            const response = await fetch('/model_info');

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const modelInfo = await response.json();

            // Perbarui tampilan dengan informasi dari server
            if (bestModelElement) bestModelElement.textContent = modelInfo.best_model;
            if (bestModelSideElement) bestModelSideElement.textContent = modelInfo.best_model;
            if (featureSetElement) featureSetElement.textContent = modelInfo.feature_set;
            if (methodUsedElement) methodUsedElement.textContent = modelInfo.feature_set;
            if (accuracyElement) accuracyElement.textContent = `${(modelInfo.accuracy * 100).toFixed(2)}%`;
            if (f1ScoreElementMain) f1ScoreElementMain.textContent = modelInfo.f1_score.toFixed(4);
            if (precisionElement) precisionElement.textContent = `${(modelInfo.precision * 100).toFixed(2)}%`;
            if (recallElement) recallElement.textContent = `${(modelInfo.recall * 100).toFixed(2)}%`;

            // Ambil semua informasi model dari server untuk perbandingan
            // Kita akan mengambil hasil dari file summary_results.csv untuk perbandingan
            // Untuk sementara, kita gunakan nilai-nilai yang diketahui dari hasil pelatihan
            if (rfeAccuracyElement) rfeAccuracyElement.textContent = '86.74%';
            if (rfeF1ScoreElement) rfeF1ScoreElement.textContent = '0.8689';
            if (allFeaturesAccuracyElement) allFeaturesAccuracyElement.textContent = '85.23%';
            if (allFeaturesF1ScoreElement) allFeaturesF1ScoreElement.textContent = '0.8517';
        } catch (error) {
            console.error('Error loading best model info:', error);
            // Gunakan nilai default jika gagal mengambil dari server
            if (bestModelElement) bestModelElement.textContent = 'RFE-RandomForest';
            if (bestModelSideElement) bestModelSideElement.textContent = 'RFE-RandomForest';
            if (featureSetElement) featureSetElement.textContent = 'RFE';
            if (methodUsedElement) methodUsedElement.textContent = 'RFE';
            if (accuracyElement) accuracyElement.textContent = '86.74%';
            if (f1ScoreElementMain) f1ScoreElementMain.textContent = '0.8689';
            if (precisionElement) precisionElement.textContent = '85.93%';
            if (recallElement) recallElement.textContent = '87.88%';

            // Isi juga data perbandingan dengan nilai default
            if (rfeAccuracyElement) rfeAccuracyElement.textContent = '86.74%';
            if (rfeF1ScoreElement) rfeF1ScoreElement.textContent = '0.8689';
            if (allFeaturesAccuracyElement) allFeaturesAccuracyElement.textContent = '85.23%';
            if (allFeaturesF1ScoreElement) allFeaturesF1ScoreElement.textContent = '0.8517';
        }
    }
    
    // Inisialisasi tampilan awal
    initializeApp();
});

// Fungsi untuk inisialisasi aplikasi
function initializeApp() {
    // Tambahkan event listener untuk input numerik agar hanya menerima angka
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            // Validasi nilai input
            const min = parseFloat(this.min);
            const max = parseFloat(this.max);
            const value = parseFloat(this.value);

            if (!isNaN(min) && value < min) this.value = min;
            if (!isNaN(max) && value > max) this.value = max;
        });
    });
}