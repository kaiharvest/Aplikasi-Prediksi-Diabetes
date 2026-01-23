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
    const featureSetElement = document.getElementById('featureSet');
    const accuracyElement = document.getElementById('accuracy');
    const f1ScoreElement = document.getElementById('f1score');
    const precisionElement = document.getElementById('precision');
    const recallElement = document.getElementById('recall');

    // Inisialisasi chart SHAP
    let shapChart = null;

    // Muat informasi model terbaik saat halaman dimuat
    loadBestModelInfo();

    // Event listener untuk formulir prediksi
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Tampilkan spinner dan nonaktifkan tombol
        btnText.classList.add('d-none');
        spinner.classList.remove('d-none');
        predictBtn.disabled = true;

        try {
            // Ambil nilai-nilai input
            const inputData = {
                pregnancies: parseFloat(document.getElementById('pregnancies').value) || 0,
                glucose: parseFloat(document.getElementById('glucose').value) || 0,
                bloodPressure: parseFloat(document.getElementById('bloodPressure').value) || 0,
                skinThickness: parseFloat(document.getElementById('skinThickness').value) || 0,
                insulin: parseFloat(document.getElementById('insulin').value) || 0,
                bmi: parseFloat(document.getElementById('bmi').value) || 0,
                dpf: parseFloat(document.getElementById('dpf').value) || 0,
                age: parseFloat(document.getElementById('age').value) || 0
            };

            // Panggil fungsi prediksi (akan diimplementasikan nanti)
            const result = await predictDiabetes(inputData);

            // Tampilkan hasil
            displayResults(result);
        } catch (error) {
            console.error('Error during prediction:', error);
            alert('Terjadi kesalahan saat melakukan prediksi. Silakan coba lagi.');
        } finally {
            // Sembunyikan spinner dan aktifkan kembali tombol
            btnText.classList.remove('d-none');
            spinner.classList.add('d-none');
            predictBtn.disabled = false;
        }
    });

    // Fungsi untuk memuat informasi model terbaik
    async function loadBestModelInfo() {
        try {
            const response = await fetch('/model_info');

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const modelInfo = await response.json();

            // Perbarui tampilan dengan informasi dari server
            bestModelElement.textContent = modelInfo.best_model;
            featureSetElement.textContent = modelInfo.feature_set;
            accuracyElement.textContent = `${(modelInfo.accuracy * 100).toFixed(2)}%`;
            f1ScoreElement.textContent = modelInfo.f1_score.toFixed(4);
            precisionElement.textContent = `${(modelInfo.precision * 100).toFixed(2)}%`;
            recallElement.textContent = `${(modelInfo.recall * 100).toFixed(2)}%`;
        } catch (error) {
            console.error('Error loading best model info:', error);
            // Gunakan nilai default jika gagal mengambil dari server
            bestModelElement.textContent = 'RFE-RandomForest';
            featureSetElement.textContent = 'RFE';
            accuracyElement.textContent = '86.74%';
            f1ScoreElement.textContent = '0.8689';
            precisionElement.textContent = '85.93%';
            recallElement.textContent = '87.88%';
        }
    }
    
    // Fungsi untuk melakukan prediksi melalui API
    async function predictDiabetes(inputData) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

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
    function displayResults(result) {
        // Sembunyikan placeholder dan tampilkan hasil
        resultPlaceholder.classList.add('d-none');
        resultContainer.classList.remove('d-none');
        
        // Tampilkan status risiko
        if (result.prediction === 1) {
            riskStatus.textContent = 'Pasien Berisiko Tinggi';
            riskStatus.className = 'h5 mb-0 font-weight-bold text-danger';
        } else {
            riskStatus.textContent = 'Pasien Berisiko Rendah';
            riskStatus.className = 'h5 mb-0 font-weight-bold text-success';
        }
        
        // Tampilkan probabilitas
        riskProbability.textContent = `${(result.probability * 100).toFixed(2)}%`;
        
        // Tampilkan informasi klinis
        updateClinicalInfo(result);
        
        // Tampilkan faktor-faktor penting
        displayFeatureImportance(result);
        
        // Gambar grafik SHAP
        drawShapChart(result);
    }
    
    // Fungsi untuk memperbarui informasi klinis
    function updateClinicalInfo(result) {
        let infoText = '';
        
        if (result.prediction === 1) {
            infoText = 'Pasien menunjukkan beberapa faktor risiko yang meningkatkan kemungkinan terkena diabetes. Disarankan untuk melakukan pemeriksaan lebih lanjut dan konsultasi dengan tenaga medis.';
        } else {
            infoText = 'Pasien memiliki faktor risiko yang relatif rendah untuk terkena diabetes. Tetap jaga pola hidup sehat untuk mencegah risiko di masa depan.';
        }
        
        clinicalDescription.textContent = infoText;
    }
    
    // Fungsi untuk menampilkan faktor-faktor penting
    function displayFeatureImportance(result) {
        // Kosongkan daftar sebelumnya
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
    
    // Fungsi untuk menggambar grafik SHAP
    function drawShapChart(result) {
        const ctx = document.getElementById('shapChart').getContext('2d');
        
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
            type: 'horizontalBar',
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
                indexAxis: 'y',
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