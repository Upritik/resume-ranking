document.getElementById('resumeForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('resume', document.getElementById('resume').files[0]);
    const jobDescription = document.getElementById('jobDescription').value;
    
    // Only append job description if it's provided
    if (jobDescription.trim() !== '') {
        formData.append('jobDescription', jobDescription);
    }

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
            // Build result HTML
            let resultHTML = `
                <p><strong>Predicted Category:</strong> ${result.category}</p>
                <p><strong>Resume Score:</strong> ${result.resume_score}%</p>
            `;

            // Only add matched skills if they exist and job description was provided
            if (jobDescription.trim() !== '' && result.matched_skills && result.matched_skills.length > 0) {
                resultHTML += `<p><strong>Matched Skills:</strong> ${result.matched_skills.join(', ')}</p>`;
            }

            document.getElementById('result').innerHTML = resultHTML;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('result').innerText = 'Failed to predict category.';
    }
});
