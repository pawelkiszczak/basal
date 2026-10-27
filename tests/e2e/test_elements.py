"""Playground screenshot element editing, serialization and evidence behavior."""
import json
import struct
import zlib

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="install basal[e2e] and run: playwright install chromium")


def png(width=1000, height=600):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    row = b"\x00" + b"\xf0\xf0\xf0" * width
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b""))


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, ui, text_model):
    ui.load(text_model["id"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    tab = context.new_page()
    errors = []
    tab.on("pageerror", lambda e: errors.append(str(e)))
    tab.goto(ui.url + "/#playground")
    tab.wait_for_function("() => typeof selectedMedia !== 'undefined' && document.querySelector('#playgroundModel').options.length")
    tab.select_option("#playgroundModel", text_model["id"])
    tab.evaluate("() => { loadedMedia=['image']; $('#mediaInput').disabled=false; $('#mediaField').hidden=false; }")
    tab.set_input_files("#mediaInput", [
        {"name": "screen.png", "mimeType": "image/png", "buffer": png()},
        {"name": "second.png", "mimeType": "image/png", "buffer": png()},
    ])
    tab.wait_for_function("() => selectedMedia.length === 2")
    yield tab
    context.close()
    assert not errors, errors



def canvas_point(page, x, y):
    box = page.locator(".elements-dialog canvas").bounding_box()
    return box["x"] + x * box["width"] / 1000, box["y"] + y * box["height"] / 600


def test_element_editor_drawing_geometry_labels_and_cleanup(page):
    source_data = page.evaluate("() => selectedMedia[0].data")
    page.locator(".mark-elements").first.click()
    canvas = page.locator(".elements-dialog canvas")
    canvas.wait_for(state="visible")
    page.wait_for_function("() => document.querySelector('.elements-dialog canvas').width === 1000")
    a, b = canvas_point(page, 30, 30), canvas_point(page, 100, 90)
    page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    page.locator(".elements-cancel").click()
    page.locator(".elements-dialog").wait_for(state="detached")
    assert page.evaluate("() => selectedMedia[0].data") == source_data
    assert page.evaluate("() => !selectedMedia[0].elements")
    page.locator(".mark-elements").first.click()
    canvas.wait_for(state="visible")
    page.wait_for_function("() => document.querySelector('.elements-dialog canvas').width === 1000")
    # The 1000px source is rendered smaller; pointer coordinates must be scaled back to natural pixels.
    tiny_a, tiny_b = canvas_point(page, 10, 10), canvas_point(page, 13, 13)
    page.mouse.move(*tiny_a); page.mouse.down(); page.mouse.move(*tiny_b); page.mouse.up()
    assert page.locator(".element-label-row").count() == 0
    for a, b in [((50, 50), (200, 150)), ((300, 50), (450, 150)), ((550, 50), (700, 150))]:
        a, b = canvas_point(page, *a), canvas_point(page, *b)
        page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    assert page.locator(".element-label-row").count() == 3
    boxes = page.evaluate("() => { const c=$('.elements-dialog canvas'),r=c.getBoundingClientRect(); return r.width < c.width; }")
    assert boxes
    page.locator('[data-label="0"]').fill("Pay now")
    page.locator('[data-label="2"]').fill("Continue")
    # Move and resize box one, then delete box two and verify compact numbering.
    first = canvas_point(page, 100, 100)
    page.mouse.move(*first); page.mouse.down(); page.mouse.move(first[0] + 15, first[1] + 12); page.mouse.up()
    corner = canvas_point(page, 220, 160)
    page.mouse.move(*corner); page.mouse.down(); page.mouse.move(corner[0] + 15, corner[1] + 15); page.mouse.up()
    page.locator(".element-label-row").nth(1).locator("[data-select]").click()
    page.keyboard.press("Delete")
    assert page.locator(".element-label-row").count() == 2
    assert [page.locator(f'[data-select="{i}"]').inner_text() for i in range(2)] == ["[1]", "[2]"]
    # Clamp a final draw at the far image edges.
    a, b = canvas_point(page, 960, 560), canvas_point(page, 1030, 640)
    page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    assert page.locator(".element-label-row").count() == 3
    page.locator("[data-label='2']").fill("Edge")
    # Exercise the hard cap by adding 23 small boxes at unused image positions.
    for i in range(23):
        x = 20 + (i % 10) * 90
        y = 220 + (i // 10) * 90
        a, b = canvas_point(page, x, y), canvas_point(page, x + 25, y + 25)
        page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    assert page.locator(".element-label-row").count() == 26
    a, b = canvas_point(page, 900, 500), canvas_point(page, 950, 550)
    page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    sync_api.expect(page.locator(".elements-error")).to_contain_text("up to 26")
    page.locator(".elements-save").click()
    page.locator(".elements-dialog").wait_for(state="detached")
    assert page.locator(".q-name").nth(1).input_value() == "element"
    assert page.locator(".q-name").count() == 2
    assert page.locator(".q-type").nth(1).input_value() == "choice"
    assert page.locator(".q-type").nth(1).is_disabled()
    assert page.locator('[data-f="criteria"]').nth(1).is_disabled()
    assert page.locator(".mark-elements").nth(1).is_hidden()
    data = page.evaluate("() => { const m=selectedMedia[0]; return {name:m.file.name,data:m.data,originalData:m.originalData,elements:m.elements,req:buildRequest()}; }")
    item, request = data, data["req"]
    assert item["name"] == "screen-marked.png"
    assert item["originalData"] and item["data"] != item["originalData"]
    assert item["elements"]["size"] == [1000, 600]
    assert len(item["elements"]["boxes"]) == 26
    assert item["elements"]["boxes"]["1"][0] > 50
    assert item["elements"]["boxes"]["1"][2] > 150
    assert item["elements"]["boxes"]["3"][0] + item["elements"]["boxes"]["3"][2] <= 1000
    assert item["elements"]["boxes"]["3"][1] + item["elements"]["boxes"]["3"][3] <= 600
    assert request["questions"]["element"]["criteria"]["1"] == "[1] Pay now"
    assert request["questions"]["element"]["criteria"]["2"] == "[2] Continue"
    assert request["questions"]["element"]["criteria"]["3"] == "[3] Edge"
    page.locator(".q-instructions-input").nth(1).fill("Choose an element")
    page.locator(".q-name").nth(1).fill("target")
    assert page.evaluate("() => selectedMedia[0].elements.question") == "target"
    assert page.evaluate("() => buildRequest().questions.target.instructions") == "Choose an element"
    # Reopening keeps the original pixels and current boxes; clear + save removes both.
    page.locator(".mark-elements").first.click()
    page.wait_for_function("() => document.querySelector('.elements-dialog canvas').width === 1000")
    assert page.locator(".element-label-row").count() == 26
    page.locator(".elements-clear").click()
    page.locator(".elements-save").click()
    assert page.evaluate("() => selectedMedia[0].data") == source_data
    # Mark again and remove its question: the screenshot is restored as well.
    page.locator(".mark-elements").first.click()
    page.wait_for_function("() => document.querySelector('.elements-dialog canvas').width === 1000")
    a, b = canvas_point(page, 40, 40), canvas_point(page, 140, 120)
    page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    page.locator(".elements-save").click()
    page.locator(".remove-question").nth(1).click()
    assert page.locator(".q-name").count() == 1
    assert page.evaluate("() => selectedMedia[0].data") == source_data
    # Removing a marked image also removes its linked question.
    page.locator(".mark-elements").first.click()
    page.wait_for_function("() => document.querySelector('.elements-dialog canvas').width === 1000")
    a, b = canvas_point(page, 40, 40), canvas_point(page, 140, 120)
    page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    page.locator(".elements-save").click()
    page.locator("[data-remove='0']").click()
    assert page.locator(".q-name").count() == 1


def test_marked_request_payload_overlay_presets_and_evidence(page):
    page.locator("#addQuestion").click()
    evidence = page.locator(".question [data-f='evidence']").first
    assert evidence.is_disabled()
    sync_api.expect(page.locator(".media-evidence-note").first).to_contain_text("state text")
    page.locator("[data-remove='1']").click()
    page.locator("[data-remove='0']").click()
    assert page.locator(".question [data-f='evidence']").first.is_enabled()
    while page.locator(".remove-question").count():
        page.locator(".remove-question").first.click()
    page.set_input_files("#mediaInput", {"name": "screen.png", "mimeType": "image/png", "buffer": png()})
    page.wait_for_function("() => selectedMedia.length === 1")

    page.locator(".mark-elements").first.click()
    page.locator(".elements-dialog canvas").wait_for(state="visible")
    page.wait_for_function("() => document.querySelector('.elements-dialog canvas').width === 1000")
    a, b = canvas_point(page, 80, 70), canvas_point(page, 230, 190)
    page.mouse.move(*a); page.mouse.down(); page.mouse.move(*b); page.mouse.up()
    page.locator("[data-label='0']").fill("Button")
    page.locator(".elements-save").click()
    # The evidence guard still rejects raw JSON carrying evidence and media.
    page.check("#editJsonToggle")
    page.evaluate("() => { const r=JSON.parse($('#requestJson').value); r.questions.element.evidence=true; $('#requestJson').value=JSON.stringify(r); }")
    calls=[]
    page.route("**/api/decide", lambda route: calls.append(route.request.post_data_json) or route.fulfill(status=200,content_type="application/json",body=json.dumps({"answers":{"element":{"type":"choice","choice":"1","probabilities":{"1":0.9}}}})))
    page.click("#sendButton")
    sync_api.expect(page.locator("#decideError")).to_contain_text("turn off")
    assert not calls
    page.evaluate("() => { const r=JSON.parse($('#requestJson').value); r.questions.element.evidence=false; $('#requestJson').value=JSON.stringify(r); }")

    request=[]
    page.route("**/api/decide", lambda route: (request.append(route.request.post_data_json),route.fulfill(status=200,content_type="application/json",body=json.dumps({"answers":{"element":{"type":"choice","choice":"1","probabilities":{"1":0.9}}}}))))
    page.click("#sendButton")
    sync_api.expect(page.locator("#results article.answer")).to_be_visible()
    assert len(request)==1
    body=request[0]
    assert body["questions"]["element"]["criteria"]=={"1":"[1] Button"}
    assert body["media"][0]["data"]==page.evaluate("() => selectedMedia[0].data")
    assert "_elements" not in body and "originalData" not in body
    svg=page.locator("#results .image-source svg")
    sync_api.expect(svg).to_be_visible()
    assert svg.locator("rect").count()==1
    assert "screen-marked.png" in svg.get_attribute("aria-label")
    original=page.evaluate("() => selectedMedia[0].originalData")
    image_data=page.evaluate("() => selectedMedia[0].data")
    pixel=page.evaluate("async data=>{const i=new Image();i.src='data:image/png;base64,'+data;await i.decode();const c=document.createElement('canvas');c.width=i.width;c.height=i.height;const x=c.getContext('2d');x.drawImage(i,0,0);return [...x.getImageData(85,40,1,1).data]}",image_data)
    assert image_data!=original and pixel[0]>180 and pixel[1]<150 and pixel[2]<100

    page.click("#saveRequestPreset")
    page.fill("#presetName","marked request")
    page.click("#presetNameSave")
    page.wait_for_function("() => presetData.request['marked request']")
    saved=page.evaluate("() => presetData.request['marked request']")
    assert saved["_elements"][0]["originalData"]==original
    assert "_elements" not in body
    page.evaluate("() => setRequest({state:'reset',questions:{other:{type:'choice',criteria:{yes:null,no:null}}}})")
    page.select_option("#examples","preset:marked request")
    assert page.locator(".q-name").first.input_value()=="element"
    assert page.evaluate("() => selectedMedia[0].originalData") == original
    assert page.evaluate("() => buildRequest().media[0].data") == image_data
    assert not page.locator("#editJsonToggle").is_checked()
    page.evaluate("() => renderResults({answers:{element:{type:'choice',choice:'1',probabilities:{'1':0.9}}}},buildRequest())")
    assert page.locator("#results .image-source svg rect").count()==1
