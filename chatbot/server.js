var http = require('http')
var spawn = require("child_process").spawn;

http.createServer(function (req, res){
    var process = spawn('python3', ["chatgui.py", req.headers.mssg.toString()]);

    process.stdout.on('data', function(data){
    res.write(data.toString());
    res.end()
    });

    

}).listen(8080)



