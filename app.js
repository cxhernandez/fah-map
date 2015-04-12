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
    fs.readdir('./static/png/', function(err, files) {

          var options = [file.match(/\d+/g)[0]];

          for (var i in files) {
            if (files[i] != file ) {
              options.push(files[i].match(/\d+/g)[0]);
            }
          }

          response.render('index', {
                          png: file,
                          options: options
          });

        });
}

app.get('/', function (req, res, next) {
    update('past30.png', res);
});

app.use('/static',express.static(__dirname + "/static/"));

app.post('/', function(req, res, next){
    update('past' + req.body.option + '.png', res);
});

app.listen(port, function () {
    console.log('fah-map listening on port ' + port);
});
