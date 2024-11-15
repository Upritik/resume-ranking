document.getElementById('resumeForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const formData = new FormData();
    formData.append('resume', document.getElementById('resume').files[0]);
    const jobDescription = document.getElementById('jobDescription').value.trim();

    if (jobDescription) {
        formData.append('jobDescription', jobDescription);
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        if (response.ok) {
            let resultHTML = `
                <p><strong>Predicted Category:</strong> ${result.category}</p>
                <p><strong>Resume Score:</strong> ${result.resume_score}%</p>
            `;
            if (result.matched_skills?.length) {
                resultHTML += `<p><strong>Matched Skills:</strong> ${result.matched_skills.join(', ')}</p>`;
            }
            if (result.missing_skills?.length) {
                resultHTML += `<p><strong>Missing Skills:</strong> ${result.missing_skills.join(', ')}</p>`;
            }
            document.getElementById('result').innerHTML = resultHTML;
        } else {
            document.getElementById('result').innerText = `Error: ${result.error || "An unknown error occurred."}`;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('result').innerText = "An error occurred while processing your request.";
    }
});
