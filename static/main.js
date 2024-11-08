document.getElementById('resumeForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('resume', document.getElementById('resume').files[0]);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();

        if (result.error) {
            document.getElementById('result').innerText = `Error: ${result.error}`;
        } else {
            document.getElementById('result').innerHTML = `
                <p><strong>Predicted Category:</strong> ${result.category}</p>
                <p><strong>Confidence Score:</strong> ${result.confidence_score}%</p>
            `;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('result').innerText = 'Failed to predict category.';
    }
});
