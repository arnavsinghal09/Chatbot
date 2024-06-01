const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  // Launch a new browser instance
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Navigate to the website
  await page.goto('https://en.wikipedia.org/wiki/Potato');

  // Example: Extract the page title
  const pageTitle = await page.title();
  console.log('Page Title:', pageTitle);

  // Example: Extract all the links (anchor tags)
  const links = await page.evaluate(() => {
    const anchorTags = Array.from(document.querySelectorAll('a'));
    return anchorTags.map(anchor => anchor.href);
  });
  console.log('Links:', links);

  // Example: Extract specific elements by class or id
  // Replace '.classname' with the actual class name or '#id' with the actual id
  const classTexts = await page.evaluate(() => {
    const classElements = Array.from(document.querySelectorAll('.classname'));
    return classElements.map(element => element.textContent);
  });
  console.log('Class Element Texts:', classTexts);

  // Extract specific elements by tag
  const h1Texts = await page.evaluate(() => {
    const h1Elements = Array.from(document.querySelectorAll('h1'));
    return h1Elements.map(element => element.textContent);
  });
  console.log('H1 Texts:', h1Texts);

  // Save data to JSON file
  const data = {
    pageTitle,
    links,
    classTexts,
    h1Texts
  };

  fs.writeFile('scraped_data.json', JSON.stringify(data, null, 2), (err) => {
    if (err) throw err;
    console.log('Data saved to scraped_data.json');
  });

  // Close the browser
  await browser.close();
})();
