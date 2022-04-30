
const img = document.querySelector('#img');
const imgShow = document.querySelector('.image-display');
const submitBtn = document.querySelector('.submit-btn');

var myImage;
img.addEventListener('change', function() {
    if (img.files[0]) {
        myImage = img.files[0];
        imgShow.src = URL.createObjectURL(myImage);
        imgShow.style.display = "block";
        document.querySelector("#pred").innerText = "Preview";
        document.querySelector("#pred").style.pointerEvents = "none";
    }   
});
