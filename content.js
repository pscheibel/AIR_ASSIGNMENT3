function msgprint() {  
            alert("You are Successfully Called the JavaScript function");  
         }  

function setAction(form) {
  form.action = "register.html";
  alert(form.action);
  return false;
}