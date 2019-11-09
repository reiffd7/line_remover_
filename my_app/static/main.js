$(document).ready(function () {
    var rotation = {
  1: 'rotate(0deg)',
  3: 'rotate(180deg)',
  6: 'rotate(90deg)',
  8: 'rotate(270deg)'
  };

  // Init
  $('.image-section').hide();
  $('.loader').hide();
  $('#result').hide();

function _arrayBufferToBase64(buffer) {
 var binary = ''
 var bytes = new Uint8Array(buffer)
 var len = bytes.byteLength;
 for (var i = 0; i < len; i++) {
   binary += String.fromCharCode(bytes[i])
 }
 return window.btoa(binary);
}
var orientation = function(file, callback) {
 var fileReader = new FileReader();
 fileReader.onloadend = function() {
   var base64img = "data:" + file.type + ";base64," + _arrayBufferToBase64(fileReader.result);
   var scanner = new DataView(fileReader.result);
   var idx = 0;
   var value = 1; // Non-rotated is the default
   if (fileReader.result.length < 2 || scanner.getUint16(idx) != 0xFFD8) {
     // Not a JPEG
     if (callback) {
       callback(base64img, value);
     }
     return;
   }
   idx += 2;
   var maxBytes = scanner.byteLength;
   while (idx < maxBytes - 2) {
     var uint16 = scanner.getUint16(idx);
     idx += 2;
     switch (uint16) {
       case 0xFFE1: // Start of EXIF
         var exifLength = scanner.getUint16(idx);
         maxBytes = exifLength - idx;
         idx += 2;
         break;
       case 0x0112: // Orientation tag
         // Read the value, its 6 bytes further out
         // See page 102 at the following URL
         // http://www.kodak.com/global/plugins/acrobat/en/service/digCam/exifStandard2.pdf
         value = scanner.getUint16(idx + 6, false);
         maxBytes = 0; // Stop scanning
         break;
     }
   }
   if (callback) {
     callback(base64img, value);
   }
 }
 fileReader.readAsArrayBuffer(file);
};



  // Upload Preview
  function readURL(input) {
      if (input.files && input.files[0]) {
          var reader = new FileReader();


          reader.onload = function (e) {
            orientation(input.files[0], function(base64img, value) {
              // $('#imagePreview').attr('src', base64img);

              $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
            console.log(value)
            console.log(rotation[value]);
              if (typeof rotation[value] !== "undefined") {
                console.log('rotated');
                $('#imagePreview').css('transform', rotation[value]);
              }
              else {

                $('#imagePreview').css('transform', rotation[1]);
              }
            });
              // $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
              // $('#imagePreview').css('transform',rotation[value]);
              $('#imagePreview').hide();
              $('#imagePreview').fadeIn(650);

            }
          reader.readAsDataURL(input.files[0]);
    }
  }
  $("#imageUpload").change(function () {
      $('.image-section').show();
      $('#btn-predict').show();
      $('#result').text('');
      $('#result').hide();
      readURL(this);
  });

  // Predict
  $('#btn-predict').click(function () {
      var form_data = new FormData($('#upload-file')[0]);

      // Show loading animation
      $(this).hide();
      $('.loader').show();

      // Make prediction by calling api /predict
      $.ajax({
          type: 'POST',
          url: '/predict',
          data: form_data,
          contentType: false,
          cache: false,
          processData: false,
          async: true,
          success: function (data) {
              // Get and display the result
              $('.loader').hide();
              $('#result').fadeIn(600);
              $('#result').html(data);
              console.log('Success!');
          },
      });
  });

});