var express = require('express');
var bodyParser = require('body-parser');
var request = require('request');
var https = require('https');
var engine = require('ejs-locals')
var fs = require('fs');

var port = process.env.PORT || 3000;

var app = express();
app.use(bodyParser());
app.engine('ejs', engine);
app.set('view engine', 'ejs');

function update(file, response) {
    fs.readdir('./static/json/', function(err, files) {
          response.render('index', {
                          json: file,
                          options: files
          });
        });
}

app.get('/', function (req, res, next) {
    update('past30.json', res);
});

app.use('/static',express.static(__dirname + "/static/"));

app.post('/', function(req, res, next){
    console.log(req.body.option)
    update(req.body.option, res);
});

app.listen(port, function () {
    console.log('fah-map listening on port ' + port);
});
