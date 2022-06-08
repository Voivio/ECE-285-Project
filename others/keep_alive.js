function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-toolbar-button").click()
}

let keepAlive = setInterval(ClickConnect,600000)

clearInterval(keepAlive)