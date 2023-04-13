import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import bodyParser from "body-parser";
import puppeteer from "puppeteer";
import fs from "fs";

const app = express();
// app.use(express.json());
app.use(bodyParser.json());
app.use(cors());

app.post("/", async (req, res) => {
  const { url } = req.body;
  // const data = Url.create({ url });
  // res.json({ data });
  res.send({ msg: url });
  // console.log(typeof url, url);
  await Puppy(url);
});

const start = async () => {
  try {
    await mongoose.connect(
      "mongodb+srv://varunasadananda34:varun@cluster0.akvhly1.mongodb.net/?retryWrites=true&w=majority"
    );
    app.listen(3000, () => {
      console.log("tatakae");
    });
  } catch (error) {
    console.log(error);
  }
};

const Puppy = async (url) => {
  if (url) {
    console.log(`"${url}"`, "why tf this is not working");
  }

  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();

  if (url) {
    await page.goto(url);

    const pageText = await page.evaluate(() => {
      return Array.from(document.querySelectorAll("body *"))
        .map((element) => element.textContent.trim().replace(/\s+/g, " "))
        .join(" ");
    });

    fs.writeFile("content.json", JSON.stringify(pageText), (err) => {
      if (err) throw err;
      console.log("Content written to file!");
    });

    // console.log(pageText);

    await browser.close();
  }
};

start();
