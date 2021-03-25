$("#formControlInputLanguage").change(function(){ //Lorsque que l'utilisateur change la langue d'entrée, le programme met l'inverse dans l'output (Exemple: entrée: Français, sortie: Anglais)
    if(document.querySelector('#formControlInputLanguage').value == "Français") document.querySelector('#formControlOutputputLanguage').selectedIndex = 1;
    else if(document.querySelector('#formControlInputLanguage').value == "Anglais") document.querySelector('#formControlOutputputLanguage').selectedIndex = 0;
})

$("#formControlOutputputLanguage").change(function(){ //Même chose mais de l'output vers l'input
    if(document.querySelector('#formControlOutputputLanguage').value == "Français") document.querySelector('#formControlInputLanguage').selectedIndex = 0;
    else if(document.querySelector('#formControlOutputputLanguage').value == "Anglais") document.querySelector('#formControlInputLanguage').selectedIndex = 1;
})

var oldVal = "";
setInterval(function(){ //Exécuté toutes les 2 secondes 
    var currentVal =document.getElementById("formControlInputLanguageTextArea").value; //Je récupère la valeur actuelle du textarea
    setTimeout(function(){ 
        if(currentVal == oldVal) return; //Si la valeut est la même, je ne fais rien
        oldVal = currentVal;
        if(document.querySelector('#formControlInputLanguage').value == "Français"){ //Si la langue d'entrée est le français (donc la sortie est en anglais) alors
            $.ajax({ //Je réalise une requête AJAX vers l'API
                url: 'http://127.0.0.1:5000/translate/', //url de l'api
                type: 'GET', //de type get
                data: { source:"fr", 
                        destination: "en",
                        original: currentVal},
                contentType: 'application/json; charset=utf-8', //L'inforamation est encapuslé en format JSON
                success: function (response) { //Si je ne reçois pas de code d'erreur alors
                    let data = response //Je copie la réponse dans une variable
                    document.getElementById("formControlOutputLanguageTextArea").value = data //Je parse la réponse pour récupérer la valeur et je la place dans le textearea de sortie
                },
                error: function (err) { //S'il y a un problème, l'erreur est émit dans la console du navigateur
                    console.log(err);
                }
            });
        }
        else if(document.querySelector('#formControlInputLanguage').value == "Anglais"){ //Même chose mais l'entrée est en anglais
            $.ajax({
                url: 'http://127.0.0.1:5000/translate/',
                type: 'GET',
                data: { source:"en", 
                        destination: "fr",
                        original: currentVal},
                contentType: 'application/json; charset=utf-8',
                success: function (response) {
                    let data = response
                    document.getElementById("formControlOutputLanguageTextArea").value = data
                },
                error: function () {
                    console.log("error");
                }
            });
        }
    }, 1000);

 }, 2000);