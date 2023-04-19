let Auth = document.getElementById("UserAuthenticationForm");
let AdvOptions = document.getElementById("AdvancedOptions");
let TestZone = document.getElementById("TestingZone");
function AuthenticateUser() {
  sessionStorage.setItem("User ID", document.forms["UserAuth"]["User"].value);
  sessionStorage.setItem("Password", document.forms["UserAuth"]["Password"].value);
  sessionStorage.setItem("TLC", document.forms["UserAuth"]["TLC"].value);
  sessionStorage.setItem("Robot", document.forms["UserAuth"]["Robot"].checked == true ? "Human" : "Robot");
  user = sessionStorage.getItem("User ID");
  pwd = sessionStorage.getItem("Password");
  TLC = sessionStorage.getItem("TLC");
  Obj = sessionStorage.getItem("Robot");

  validationDetails();
}
var user = sessionStorage.getItem("User ID");
var pwd = sessionStorage.getItem("Password");
var TLC = sessionStorage.getItem("TLC");
var Obj = sessionStorage.getItem("Robot");
var data = JSON.parse(sessionStorage.getItem("Data"));
var Disable = sessionStorage.getItem("Disable");
var Product = sessionStorage.getItem("ProductSerial");
var AdvOpt = sessionStorage.getItem("AdvSearch");
var BegProcess = sessionStorage.getItem("BeginSearch");
function validationDetails() {
  if (TLC == "NO TLC") {
    fetch("./templates/FullAccess.json")
      .then((response) => {
        return response.json();
      })
      .then((data) => {
        let Json = data;
        sessionStorage.setItem("Data", JSON.stringify(Json));
      })
      .catch(function (error) {
        console.log(error);
      });
    var data = JSON.parse(sessionStorage.getItem("Data"));
    var status = 0;
    for (x in data) {
      var USER = data[x]["_User_Id"];
      var PSWD = data[x]["_Code"];
      if ("TESTCASE01" == user && "TESTCASE01" == pwd) {
        sessionStorage.setItem("Disable", 1);
        status = 1;
        var Name = data[x]["_Name"];
        break;
      }
    }
    status == 0 ? alert("No User Found") : confirm(" Welcome, " + Name);
  }
}
if (Disable == 1) {
  Auth.style.display = "none";
  TestZone.style.display = "block";
}
function ProductSearch() {
  sessionStorage.setItem("ProductSerial", document.forms["Testing"]["ProductId"].value);
  sessionStorage.setItem("BeginSearch", document.forms["Testing"]["BeginProcess"].checked == true ? 1 : 0);
  sessionStorage.setItem("AdvSearch", document.forms["Testing"]["AvdSearch"].checked == true ? 1 : 0);
  Product = sessionStorage.getItem("ProductSerial");
  AdvOpt = sessionStorage.getItem("AdvSearch");
  BegProcess = sessionStorage.getItem("BeginSearch");
  if (AdvOpt == 1) {
    console.log("Adv Option");
  }
}
if (BegProcess == 1) {
  location.href = "result.html";
} else {
  location.href = "#";
}
function time() {
  const d = new Date();
  let h = addZero(d.getHours());
  let m = addZero(d.getMinutes());
  let s = addZero(d.getSeconds());
  let time = h + ":" + m + ":" + s;
  return time;
}
function date() {
  const d = new Date();
  let date = addZero(d.getDate());
  let month = addZero(d.getMonth() + 1);
  let year = addZero(d.getFullYear());
  let today = date + "-" + month + "-" + year;
  return today;
}
