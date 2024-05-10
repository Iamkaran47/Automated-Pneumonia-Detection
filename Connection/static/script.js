function predictImage() {
    const input = document.getElementById('uploadInput');
    const file = input.files[0];

    if (file) {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('predictionResult').innerText = `Predicted Class: ${data.prediction}`;
            document.getElementById('accuracyResult').innerText = `Accuracy: ${data.accuracy.toFixed(2)}%`;
        })
        .catch(error => console.error(error));
    }
}
