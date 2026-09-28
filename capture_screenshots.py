import asyncio
import time
from playwright.async_api import async_playwright

async def run_scenarios():
    print("Starting screenshot capture...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        scenarios = [
            ("01_single_vqa", 0),
            ("02_water_grounding", 1),
            ("03_temporal_change", 2),
            ("04_optical_sar", 3),
            ("05_hero", 4)
        ]

        for name, idx in scenarios:
            print(f"Running scenario: {name}")
            page = await browser.new_page()
            page.set_default_timeout(180000) # 3 minutes timeout for heavy inference
            await page.set_viewport_size({"width": 1440, "height": 1080})
            
            try:
                await page.goto("http://localhost:8000")
                await page.wait_for_selector("#submit-btn")
                
                # Select the demo query
                await page.evaluate(f"app.setDemoQuery({idx})")
                await page.wait_for_timeout(500)
                
                # Click execute
                await page.click("#submit-btn")
                
                # Wait for loading overlay to be visible then hidden
                try:
                    await page.wait_for_selector("#loading-overlay", state="visible", timeout=5000)
                except:
                    pass # Might be too fast or already hidden
                
                print("Waiting for execution to finish...")
                await page.wait_for_selector("#loading-overlay", state="hidden", timeout=180000)
                
                # Extra wait for UI to render map and panels fully
                await page.wait_for_timeout(3000)
                
                # Expand any execution trace if closed (trace-content inside trace-panel might be expanded)
                # But it's fine, whatever the default UI looks like is realistic.
                
                screenshot_path = f"final_validation/{name}.png"
                await page.screenshot(path=screenshot_path, full_page=True)
                print(f"Saved {screenshot_path}")
                
            except Exception as e:
                print(f"Error on {name}: {e}")
            
            finally:
                await page.close()

        await browser.close()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(run_scenarios())
