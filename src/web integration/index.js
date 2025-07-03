// Send given array to Python flask server.
const sendStringsToPython = async (array) => {
    
    const response = await fetch("http://127.0.0.1:5000/send_strings", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ strings: array })
    });

    const data = await response.json();
    console.log("Response from Python:", data);
    let pred = data.received;
    console.log(pred);
    if (pred[0] == 0) {
        document.getElementById('result').textContent = "Model predicted RUN with " + pred[1] + "% certainty.";
    } else if (pred[0] == 1) {
        document.getElementById('result').textContent = "Model predicted PASS with " + pred[1] + "% certainty.";
    } else {
        document.getElementById('result').textContent = "Received an unexpected result";
    }
};

$(document).ready(function() {
    // When Predict button is clicked, collects contents of all boxes and begins sending to Python flask server.
    $("#collect").click(function() {
        console.log("Button clicked!")
        let array = $(".predictor").map(function() {
            return $(this).val();
        }).get();
        console.log(array);
        sendStringsToPython(array);
    });
});