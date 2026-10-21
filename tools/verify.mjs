import { chromium } from 'playwright';

const [, , base, flower, outPrefix] = process.argv;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const errors = [];
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
page.on('pageerror', (e) => errors.push(String(e)));
page.on('requestfailed', (r) => errors.push('failed ' + r.url()));

const t0 = Date.now();
await page.goto(base + '?flower=' + flower);
await page.waitForFunction(() => document.querySelector('.bloom').classList.contains('ready'), null, { timeout: 30000 });
console.log('frames loaded after', Date.now() - t0, 'ms');

const sig = () => page.evaluate(() => {
  const c = document.querySelector('.bloom');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let s = 0;
  for (let i = 0; i < d.length; i += 97) s = (s * 31 + d[i]) % 1e9;
  return s;
});

const start = Date.now();
for (const ms of [50, 1100, 2200, 3600, 5800]) {
  await page.waitForTimeout(Math.max(0, ms - (Date.now() - start)));
  await page.screenshot({ path: `${outPrefix}-${ms}.png` });
}
const a = await sig();
await page.waitForTimeout(1500);
const b = await sig();
console.log('still after end:', a === b);
console.log('errors:', errors.length ? errors : 'none');
await browser.close();
