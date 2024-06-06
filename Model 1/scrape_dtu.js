
const puppeteer = require('puppeteer');
const fs = require('fs');

(async () => {
  // Launch a new browser instance
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  // Navigate to the website
  await page.goto('https://www.dtu.ac.in/');

  // Increase the timeout and wait for a specific element to load
  const selector = '.colr'; // Adjust the selector
  const timeout = 10000; // Set the timeout to 10 seconds

  try {
    await page.waitForSelector(selector, { timeout });
  } catch (error) {
    console.error(`Error: ${error.message}`);
    await browser.close();
    return;
  }

  // Extract main text content
  const mainText = await page.evaluate(() => {
    const content = document.querySelector('.colr'); // Adjust the selector to match the main content
    return content ? content.textContent.trim() : '';
  });
  console.log('Main Text:', mainText);

  // Extract specific elements by class or id
  const classTexts = await page.evaluate(() => {
    const classElements = Array.from(document.querySelectorAll('.colr')); // Adjust selector as needed
    return classElements.map(element => element.textContent.trim());
  });
  console.log('Class Element Texts:', classTexts);

  // Extract specific elements by tag
  const h1Texts = await page.evaluate(() => {
    const h1Elements = Array.from(document.querySelectorAll('h1'));
    return h1Elements.map(element => element.textContent.trim());
  });
  console.log('H1 Texts:', h1Texts);

  // Save data to JSON file
  const data = {
    mainText,
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
