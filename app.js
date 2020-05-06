let express = require("express");
let fs = require("fs");

const app = express();
const port = 5000;

app.use(express.static("./"));
app.use(express.urlencoded());

app.get("/", function(req, res) {
  res.send("index");
});

app.listen(port, () => console.log(`Listening on port ${port}`));
