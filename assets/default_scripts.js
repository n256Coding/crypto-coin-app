console.log("Hello");

// (document).ready(function () {
//     console.log("Hello");
//     $("#btn-clustering").click(function () {
//         $(this).hide();
//     });
// });

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        large_params_function: function() {
            return someTransform(largeValue1, largeValue2);
        }
    }
});


// document.addEventListener('DOMContentLoaded', () => {
//     const menuIcon = document.querySelector('#btn-clustering');
//     // const navbar = document.querySelector('.navbar');
  
//     console.log(menuIcon);
//     // console.log(navbar);
  
//     // menuIcon.onclick = () => {
//     //   menuIcon.classList.toggle('bx-x');
//     //   navbar.classList.toggle('active');
//     // };
//   });