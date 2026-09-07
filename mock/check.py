"""Prototype browser checks. Run against the static server; no production access."""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:4173')
parser.add_argument('--browser', default=None)
parser.add_argument('--screenshots', default='/tmp/courtside-preview')
args = parser.parse_args()
screenshots = Path(args.screenshots)
screenshots.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=args.browser)
    page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    external = []
    page.on('request', lambda request: external.append(request.url) if not request.url.startswith(args.url) else None)
    checks = 0
    for width in [360, 390, 768, 1024, 1440]:
        page.set_viewport_size({'width': width, 'height': 900 if width > 760 else 844})
        for theme in ['light', 'dark']:
            for route in ['overview', 'players', 'player/1', 'record', 'signup', 'trends']:
                page.goto(f'{args.url}/#{route}')
                page.wait_for_load_state('networkidle')
                page.evaluate('(theme) => setTheme(theme)', theme)
                page.locator('h1').wait_for()
                overflow = page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
                assert not overflow, f'Horizontal overflow: {width}, {theme}, {route}'
                if width in [390, 1440] and route in ['overview', 'player/1', 'record', 'signup', 'trends']:
                    page.screenshot(path=str(screenshots / f'{route.replace("/", "-")}-{width}-{theme}.png'), full_page=True, animations='disabled')
                checks += 1
    page.goto(f'{args.url}/#overview')
    expect(page.locator('#weekly-quote')).to_be_visible()
    quote = page.locator('#weekly-quote').inner_text()
    page.locator('[data-community="quote-prev"]').click()
    assert page.locator('#weekly-quote').inner_text() != quote
    page.locator('[data-community="quote-next"]').click()
    assert page.locator('#weekly-quote').inner_text() == quote
    page.reload()
    assert page.locator('#weekly-quote').inner_text() == quote
    assert page.evaluate("weeklyQuote(new Date('2026-09-06T16:00:00Z'), 0).key") == '2026-09-07'
    assert page.evaluate("weeklyQuote(new Date('2026-09-06T15:59:59Z'), 0).key") == '2026-08-31'
    assert page.locator('.club-line').count() == 8
    page.goto(f'{args.url}/#trends')
    expect(page.locator('#club-chart-svg')).to_be_visible()
    page.locator('[data-community="select-none"]').click()
    assert page.locator('.club-line').count() == 0
    expect(page.locator('.empty-trend')).to_be_visible()
    page.locator('[data-community="select-me"]').click()
    assert page.locator('.club-line').count() == 1
    page.locator('[data-community="toggle-player"][data-id="2"]').click()
    assert page.locator('.club-line').count() == 2
    page.locator('[data-community="select-all"]').click()
    assert page.locator('.club-line').count() == 8
    page.locator('[data-community="trend-range"][data-value="4"]').click()
    assert page.locator('.club-date-hit').count() == 5
    page.locator('.club-date-hit').first.focus()
    assert page.locator('#club-readout-date').inner_text() == '2026-08-10'
    ratings = page.locator('.club-value strong').all_text_contents()
    assert list(map(int, ratings)) == sorted(map(int, ratings), reverse=True)
    page.locator('[data-community="trend-mode"][data-value="rank"]').click()
    expect(page.locator('#club-chart-svg')).to_contain_text('#1')
    assert page.locator('#club-readout-date').inner_text() == '2026-08-10'
    page.set_viewport_size({'width': 390, 'height': 844})
    page.goto(f'{args.url}/#signup')
    expect(page.locator('h1')).to_have_text('每周报名')
    initial_people = int(page.locator('[data-signup-total]').inner_text())
    initial_rows = page.locator('.roster-row').count()
    page.locator('[data-community="party-size"][data-value="2"]').click()
    page.locator('[data-community="join-signup"]').click()
    assert int(page.locator('[data-signup-total]').inner_text()) == initial_people + 2
    assert page.locator('.roster-row').count() == initial_rows + 1
    page.locator('[data-community="party-size"][data-value="1"]').click()
    assert int(page.locator('[data-signup-total]').inner_text()) == initial_people + 1
    page.screenshot(path=str(screenshots / 'signup-joined-390-dark.png'), full_page=True, animations='disabled')
    page.goto(f'{args.url}/#overview')
    expect(page.locator('.signup-preview')).to_contain_text('已报名')
    assert int(page.locator('[data-signup-total]').inner_text()) == initial_people + 1
    page.goto(f'{args.url}/#signup')
    page.locator('[data-community="cancel-signup"]').click()
    page.get_by_role('button', name='保留报名', exact=True).click()
    assert int(page.locator('[data-signup-total]').inner_text()) == initial_people + 1
    page.locator('[data-community="cancel-signup"]').click()
    page.locator('[data-community="confirm-cancel"]').click()
    assert int(page.locator('[data-signup-total]').inner_text()) == initial_people
    assert page.locator('.roster-row').count() == initial_rows
    page.locator('.signup-identity [data-action="identity"]').click()
    page.get_by_role('button', name='阿哲', exact=True).click()
    expect(page.locator('.signup-confirmed')).to_contain_text('已报名 · 1 人参加')
    page.locator('.signup-identity [data-action="identity"]').click()
    page.get_by_role('button', name='林一', exact=True).click()
    expect(page.locator('[data-community="join-signup"]')).to_be_visible()
    page.reload()
    assert int(page.locator('[data-signup-total]').inner_text()) == initial_people
    page.set_viewport_size({'width': 390, 'height': 844})
    page.goto(f'{args.url}/#players')
    page.locator('#player-search').fill('小雨')
    assert page.locator('.player-card').count() == 1
    page.locator('.player-card').click()
    expect(page.locator('h1')).to_contain_text('小雨')
    page.get_by_role('button', name='近 12 周', exact=True).click()
    assert page.locator('.chart-hit').count() > 8
    page.get_by_role('button', name='失利', exact=True).click()
    assert page.locator('.match-row .result-dot:not(.loss)').count() == 0
    count = page.locator('.match-row').count()
    page.locator('[data-action="load-more"]').click()
    assert page.locator('.match-row').count() > count
    page.locator('[data-action="relations"][data-kind="partners"]').click()
    assert page.locator('dialog').is_visible()
    page.keyboard.press('Escape')
    assert not page.locator('dialog').is_visible()
    page.evaluate("setTheme('light')")
    page.get_by_role('button', name='切换深色模式', exact=True).click()
    page.reload()
    assert page.locator('html').get_attribute('data-theme') == 'dark'
    page.goto(f'{args.url}/#record')
    assert page.locator('#submit-record').is_disabled()
    for index, name in enumerate(['林一', '阿哲', '小雨', '嘉宁']):
        page.locator(f'[data-action="slot"][data-index="{index}"]').click()
        if index:
            assert page.locator('[data-action="pick"][data-id="1"]').is_disabled()
        page.locator('[data-input="picker-search"]').fill(name)
        page.get_by_role('button', name=name, exact=True).click()
    page.get_by_role('spinbutton', name='B队比分').fill('21')
    assert page.locator('#submit-record').is_disabled()
    page.get_by_role('spinbutton', name='B队比分').fill('18')
    assert page.locator('#submit-record').is_enabled()
    page.evaluate('window.scrollTo(0, 0)')
    submit_box = page.locator('#submit-record').bounding_box()
    nav_box = page.locator('.bottom-nav').bounding_box()
    assert submit_box['y'] + submit_box['height'] <= nav_box['y'], 'Primary record action hidden behind navigation'
    page.screenshot(path=str(screenshots / 'record-selected-390-dark.png'), full_page=True, animations='disabled')
    page.get_by_role('spinbutton', name='B队比分').fill('100')
    assert page.locator('#submit-record').is_disabled()
    page.get_by_role('spinbutton', name='B队比分').fill('18')
    page.get_by_role('button', name='B队加一分', exact=True).click()
    assert page.get_by_role('spinbutton', name='B队比分').input_value() == '19'
    page.get_by_role('button', name='交换两边', exact=True).click()
    assert page.get_by_role('spinbutton', name='A队比分').input_value() == '19'
    assert '小雨' in page.locator('[data-action="slot"][data-index="0"]').inner_text()
    before = page.evaluate('MATCHES.length')
    page.locator('#submit-record').click()
    page.get_by_role('button', name='返回修改', exact=True).click()
    assert page.evaluate('MATCHES.length') == before
    fresh_context = browser.new_context(color_scheme='dark')
    fresh_page = fresh_context.new_page()
    fresh_page.goto(args.url)
    assert fresh_page.locator('html').get_attribute('data-theme') == 'dark'
    fresh_page.emulate_media(color_scheme='light')
    expect(fresh_page.locator('html')).to_have_attribute('data-theme', 'light')
    fresh_context.close()
    page.locator('#submit-record').click()
    page.get_by_role('button', name='确认 · 演示记分', exact=True).click()
    assert page.evaluate('MATCHES.length') == before + 1
    assert page.locator('.delta-row').count() == 4
    page.screenshot(path=str(screenshots / 'record-success-390-dark.png'), full_page=True, animations='disabled')
    page.get_by_role('button', name='再记一场', exact=False).click()
    assert page.locator('.slot:not(.empty)').count() == 4
    assert page.get_by_role('spinbutton', name='A队比分').input_value() == '21'
    page.reload()
    assert page.evaluate('MATCHES.length') == before
    assert not external, f'Unexpected external requests: {external}'
    assert not errors, f'Browser errors: {errors}'
    print(f'PASS: {checks} viewport/theme/page combinations; weekly quote stability and Shanghai week boundary; all-player trends, selection, dates and ranks; signup, party size, cancellation, identity and homepage sync; search, filters, theme persistence, score validation, swap, ELO feedback and reload isolation.')
    print(f'Screenshots: {screenshots}')
    browser.close()
