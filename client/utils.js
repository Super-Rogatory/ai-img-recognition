// function handles manipulating the DOM to display a popup
function showMoreInformation() {
    var popup = document.getElementById("detailsPopup");
    popup.classList.toggle("show");
}

// allows us to preview our files in the image window screen
function previewFile() {
    var preview = document.querySelector('img');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.onloadend = function () {
        preview.src = reader.result;
    }

    if (file) {
        reader.readAsDataURL(file);
    } else {
        preview.src = "";
    }
}

async function getResultsFromAI() {
    // if user changes the port number in the .env file this will no longer work!
    var formData = new FormData();
    var fileInput = document.getElementById('file');
    formData.append("file", fileInput.files[0]);
    const res = await fetch('http://localhost:5000/checkpicture', {
        method: 'POST',
        body: formData,
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'multipart/form-data'
        },

    })
    // const res = await fetch(`http://localhost:5000/checkpicture`, {
    //     method: 'POST',
    //     body: formData
    // })

}