# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

out_dir = Path(r'C:\SignProject\Paper\figures')
out_dir.mkdir(parents=True, exist_ok=True)

W, H = 1800, 1000
img = Image.new('RGB', (W, H), (250, 252, 255))
d = ImageDraw.Draw(img)

font_regular = str(Path(r'C:\Windows\Fonts\malgun.ttf'))
font_bold = str(Path(r'C:\Windows\Fonts\malgunbd.ttf'))

title_font = ImageFont.truetype(font_bold, 40)
section_font = ImageFont.truetype(font_bold, 28)
section_long_font = ImageFont.truetype(font_bold, 22)
body_font = ImageFont.truetype(font_regular, 22)
small_font = ImageFont.truetype(font_regular, 18)

navy = (32, 57, 95)
blue = (71, 117, 214)
light_blue = (229, 238, 255)
gray = (90, 98, 108)
light_gray = (241, 244, 248)
dark = (34, 37, 43)
green = (74, 156, 96)
orange = (214, 140, 71)
purple = (113, 92, 176)
line = (170, 180, 194)

TITLE = '실시간 수어 인식 및 번역 시스템의 전체 구성도'
d.text((60, 35), TITLE, fill=dark, font=title_font)
d.rounded_rectangle((60, 90, 1740, 96), radius=3, fill=blue)

def box(x1, y1, x2, y2, title, lines, fill, outline, title_fill=(255,255,255), title_font_override=None):
    d.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline=outline, width=3)
    d.rounded_rectangle((x1, y1, x2, y1+48), radius=18, fill=outline)
    d.rectangle((x1, y1+24, x2, y1+48), fill=outline)
    tf = title_font_override or section_font
    d.text((x1+18, y1+9), title, fill=title_fill, font=tf)
    yy = y1 + 70
    for line_text in lines:
        d.text((x1+18, yy), line_text, fill=dark, font=body_font)
        yy += 34

def arrow(x1, y1, x2, y2, color=navy, width=6, label=None):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2-y1, x2-x1)
    ah = 16
    aw = 9
    p1 = (x2, y2)
    p2 = (x2 - ah*math.cos(ang) + aw*math.sin(ang), y2 - ah*math.sin(ang) - aw*math.cos(ang))
    p3 = (x2 - ah*math.cos(ang) - aw*math.sin(ang), y2 - ah*math.sin(ang) + aw*math.cos(ang))
    d.polygon([p1,p2,p3], fill=color)
    if label:
        tx = (x1+x2)//2
        ty = (y1+y2)//2 - 24
        bbox = d.textbbox((0,0), label, font=small_font)
        tw = bbox[2]-bbox[0]
        th = bbox[3]-bbox[1]
        d.rounded_rectangle((tx-tw//2-10, ty-6, tx+tw//2+10, ty+th+6), radius=10, fill=(255,255,255), outline=line)
        d.text((tx-tw//2, ty), label, fill=gray, font=small_font)

box(80, 150, 500, 360, '입력 1. 비전 데이터', [
    '웹캠 영상 입력',
    '양손 수어 동작 촬영',
    '실시간 프레임 획득'
], light_blue, blue)

box(80, 430, 500, 720, '입력 2. 스마트 장갑 데이터', [
    '좌/우 손 Flex 센서',
    '좌/우 손 IMU 센서',
    'Arduino Mega 기반 센서 수집',
    '시리얼 통신으로 PC 전송'
], (236, 249, 239), green)

box(620, 150, 1060, 360, '비전 특징 추출', [
    'MediaPipe Hands 적용',
    '손 랜드마크 추출',
    '좌/우 손 추적 및 슬롯 할당',
    '60프레임 시계열 구성'
], (238, 243, 255), navy)

box(620, 430, 1060, 720, '센서 특징 구성', [
    'IMU: temporal sequence 유지',
    'Flex: posture-summary feature 변환',
    '기준 프레임 대비 변화량 계산',
    '비전 프레임과 시간축 정렬'
], (255, 245, 233), orange)

box(1180, 240, 1600, 520, 'Feature Redesigned Fusion 기반 LSTM', [
    'vision sequence',
    'IMU / flag sequence',
    'flex posture feature',
    '멀티모달 융합 및 시계열 학습'
], (244, 239, 255), purple, title_font_override=section_long_font)

box(1180, 600, 1600, 820, '출력', [
    '단어별 신뢰도 산출',
    '최고 신뢰도 단어 선택',
    '실시간 수어 번역 결과 표시'
], light_gray, gray)

arrow(500, 255, 620, 255, label='영상 처리')
arrow(500, 575, 620, 575, label='센서 처리')
arrow(1060, 255, 1180, 340, label='비전 특징')
arrow(1060, 575, 1180, 420, label='센서 특징')
arrow(1390, 520, 1390, 600, label='softmax\n신뢰도')

note_y1, note_y2 = 865, 960
d.rounded_rectangle((80, note_y1, 1720, note_y2), radius=16, fill=(252, 252, 252), outline=line, width=2)
d.text((105, 888), '그림 설명: 웹캠 비전 데이터와 장갑 센서 데이터를 동시에 수집한 뒤, MediaPipe Hands와 센서 특징 재구성을 거쳐', fill=gray, font=body_font)
d.text((105, 922), 'Feature Redesigned Fusion 기반 LSTM 모델에 입력하여 최종 단어 신뢰도를 계산하고 실시간 번역 결과를 출력한다.', fill=gray, font=body_font)

png_path = out_dir / 'fig_system_overview.png'
img.save(png_path)

svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1000" viewBox="0 0 1800 1000">
  <rect width="1800" height="1000" fill="#fafcff"/>
  <text x="60" y="70" font-size="40" font-family="Malgun Gothic, sans-serif" font-weight="700" fill="#22252b">실시간 수어 인식 및 번역 시스템의 전체 구성도</text>
  <rect x="60" y="90" width="1680" height="6" rx="3" fill="#4775d6"/>
  <rect x="80" y="150" width="420" height="210" rx="18" fill="#e5eeff" stroke="#4775d6" stroke-width="3"/>
  <rect x="80" y="150" width="420" height="48" rx="18" fill="#4775d6"/>
  <text x="98" y="182" font-size="28" font-family="Malgun Gothic, sans-serif" font-weight="700" fill="#ffffff">입력 1. 비전 데이터</text>
  <text x="98" y="238" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">웹캠 영상 입력</text>
  <text x="98" y="272" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">양손 수어 동작 촬영</text>
  <text x="98" y="306" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">실시간 프레임 획득</text>

  <rect x="80" y="430" width="420" height="290" rx="18" fill="#ecf9ef" stroke="#4a9c60" stroke-width="3"/>
  <rect x="80" y="430" width="420" height="48" rx="18" fill="#4a9c60"/>
  <text x="98" y="462" font-size="28" font-family="Malgun Gothic, sans-serif" font-weight="700" fill="#ffffff">입력 2. 스마트 장갑 데이터</text>
  <text x="98" y="518" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">좌/우 손 Flex 센서</text>
  <text x="98" y="552" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">좌/우 손 IMU 센서</text>
  <text x="98" y="586" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">Arduino Mega 기반 센서 수집</text>
  <text x="98" y="620" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">시리얼 통신으로 PC 전송</text>

  <rect x="620" y="150" width="440" height="210" rx="18" fill="#eef3ff" stroke="#20395f" stroke-width="3"/>
  <rect x="620" y="150" width="440" height="48" rx="18" fill="#20395f"/>
  <text x="638" y="182" font-size="28" font-family="Malgun Gothic, sans-serif" font-weight="700" fill="#ffffff">비전 특징 추출</text>
  <text x="638" y="238" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">MediaPipe Hands 적용</text>
  <text x="638" y="272" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">손 랜드마크 추출</text>
  <text x="638" y="306" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">좌/우 손 추적 및 슬롯 할당</text>
  <text x="638" y="340" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">60프레임 시계열 구성</text>

  <rect x="620" y="430" width="440" height="290" rx="18" fill="#fff5e9" stroke="#d68c47" stroke-width="3"/>
  <rect x="620" y="430" width="440" height="48" rx="18" fill="#d68c47"/>
  <text x="638" y="462" font-size="28" font-family="Malgun Gothic, sans-serif" font-weight="700" fill="#ffffff">센서 특징 구성</text>
  <text x="638" y="518" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">IMU: temporal sequence 유지</text>
  <text x="638" y="552" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">Flex: posture-summary feature 변환</text>
  <text x="638" y="586" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">기준 프레임 대비 변화량 계산</text>
  <text x="638" y="620" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">비전 프레임과 시간축 정렬</text>

  <rect x="1180" y="240" width="420" height="280" rx="18" fill="#f4efff" stroke="#715cb0" stroke-width="3"/>
  <rect x="1180" y="240" width="420" height="48" rx="18" fill="#715cb0"/>
  <text x="1198" y="270" font-size="22" font-family="Malgun Gothic, sans-serif" font-weight="700" fill="#ffffff">Feature Redesigned Fusion 기반 LSTM</text>
  <text x="1198" y="328" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">vision sequence</text>
  <text x="1198" y="362" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">IMU / flag sequence</text>
  <text x="1198" y="396" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">flex posture feature</text>
  <text x="1198" y="430" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">멀티모달 융합 및 시계열 학습</text>

  <rect x="1180" y="600" width="420" height="220" rx="18" fill="#f1f4f8" stroke="#5a626c" stroke-width="3"/>
  <rect x="1180" y="600" width="420" height="48" rx="18" fill="#5a626c"/>
  <text x="1198" y="632" font-size="28" font-family="Malgun Gothic, sans-serif" font-weight="700" fill="#ffffff">출력</text>
  <text x="1198" y="688" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">단어별 신뢰도 산출</text>
  <text x="1198" y="722" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">최고 신뢰도 단어 선택</text>
  <text x="1198" y="756" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#22252b">실시간 수어 번역 결과 표시</text>

  <line x1="500" y1="255" x2="620" y2="255" stroke="#20395f" stroke-width="6"/>
  <polygon points="620,255 604,246 604,264" fill="#20395f"/>
  <line x1="500" y1="575" x2="620" y2="575" stroke="#20395f" stroke-width="6"/>
  <polygon points="620,575 604,566 604,584" fill="#20395f"/>
  <line x1="1060" y1="255" x2="1180" y2="340" stroke="#20395f" stroke-width="6"/>
  <polygon points="1180,340 1164,332 1170,347" fill="#20395f"/>
  <line x1="1060" y1="575" x2="1180" y2="420" stroke="#20395f" stroke-width="6"/>
  <polygon points="1180,420 1166,431 1181,438" fill="#20395f"/>
  <line x1="1390" y1="520" x2="1390" y2="600" stroke="#20395f" stroke-width="6"/>
  <polygon points="1390,600 1381,584 1399,584" fill="#20395f"/>

  <rect x="80" y="865" width="1640" height="95" rx="16" fill="#fcfcfc" stroke="#aab4c2" stroke-width="2"/>
  <text x="105" y="910" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#5a626c">그림 설명: 웹캠 비전 데이터와 장갑 센서 데이터를 동시에 수집한 뒤, MediaPipe Hands와 센서 특징 재구성을 거쳐</text>
  <text x="105" y="944" font-size="22" font-family="Malgun Gothic, sans-serif" fill="#5a626c">Feature Redesigned Fusion 기반 LSTM 모델에 입력하여 최종 단어 신뢰도를 계산하고 실시간 번역 결과를 출력한다.</text>
</svg>'''
(out_dir / 'fig_system_overview.svg').write_text(svg, encoding='utf-8')
print('restored')
