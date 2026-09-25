#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "deliverables");
const ASSETS = path.join(OUT, "doc_assets");
fs.mkdirSync(OUT, { recursive: true });

const C = {
  navy: "162738",
  blue: "2C5F7C",
  gold: "C79A3B",
  green: "25836B",
  red: "B94A48",
  ink: "23303B",
  muted: "66737E",
  line: "D9E0E5",
  pale: "F2F5F7",
  paleGold: "F7F0E2",
  paleBlue: "EAF1F5",
  paleGreen: "E7F2EF",
  white: "FFFFFF",
};

const PAGE_WIDTH = 11906;
const PAGE_HEIGHT = 16838;
const CONTENT_WIDTH = 10100;
const DOC_FONT = {
  ascii: "Arial",
  hAnsi: "Arial",
  eastAsia: "Hiragino Sans GB",
  cs: "Arial",
};

function run(text, options = {}) {
  return new TextRun({
    text: String(text),
    font: options.font || DOC_FONT,
    size: options.size || 21,
    bold: !!options.bold,
    italics: !!options.italics,
    color: options.color || C.ink,
    break: options.break,
  });
}

function paragraph(textOrRuns = "", options = {}) {
  const children = Array.isArray(textOrRuns)
    ? textOrRuns
    : [run(textOrRuns, options.run || {})];
  return new Paragraph({
    children,
    alignment: options.alignment || AlignmentType.LEFT,
    spacing: {
      before: options.before || 0,
      after: options.after === undefined ? 110 : options.after,
      line: options.line || 330,
    },
    keepNext: !!options.keepNext,
    keepLines: options.keepLines === undefined ? true : !!options.keepLines,
    pageBreakBefore: !!options.pageBreakBefore,
    indent: options.indent,
    border: options.border,
  });
}

function heading(text, level = 1, options = {}) {
  const map = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3,
  };
  return new Paragraph({
    text,
    heading: map[level],
    keepNext: true,
    keepLines: true,
    pageBreakBefore: !!options.pageBreakBefore,
    spacing: {
      before: level === 1 ? 280 : 180,
      after: level === 1 ? 150 : 100,
    },
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    children: [run(text)],
    numbering: { reference: "bullets", level },
    spacing: { after: 70, line: 320 },
    keepLines: true,
  });
}

function numbered(text, level = 0) {
  return new Paragraph({
    children: [run(text)],
    numbering: { reference: "numbers", level },
    spacing: { after: 80, line: 320 },
    keepLines: true,
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function cell(content, width, options = {}) {
  const children = Array.isArray(content)
    ? content
    : [
        paragraph(content, {
          alignment: options.alignment || AlignmentType.LEFT,
          after: 20,
          line: 280,
          run: {
            size: options.fontSize || 18,
            bold: !!options.bold,
            color: options.color || C.ink,
          },
        }),
      ];
  return new TableCell({
    children,
    width: { size: width, type: WidthType.DXA },
    verticalAlign: options.verticalAlign || VerticalAlign.CENTER,
    shading: options.shading
      ? { type: ShadingType.CLEAR, fill: options.shading, color: "auto" }
      : undefined,
    margins: {
      top: options.margin || 100,
      bottom: options.margin || 100,
      left: options.margin || 110,
      right: options.margin || 110,
    },
  });
}

function table(headers, rows, widths, options = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    cantSplit: true,
    tableHeader: true,
    children: headers.map((value, index) =>
      cell(value, widths[index], {
        bold: true,
        color: C.white,
        shading: options.headerColor || C.navy,
        alignment: options.alignments?.[index] || AlignmentType.LEFT,
        fontSize: options.fontSize || 17,
      })
    ),
  });
  const bodyRows = rows.map(
    (row, rowIndex) =>
      new TableRow({
        cantSplit: true,
        children: row.map((value, index) =>
          cell(value, widths[index], {
            shading: rowIndex % 2 === 0 ? C.white : C.pale,
            alignment: options.alignments?.[index] || AlignmentType.LEFT,
            fontSize: options.fontSize || 17,
          })
        ),
      })
  );
  return new Table({
    rows: [headerRow, ...bodyRows],
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      left: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      right: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      insideVertical: { style: BorderStyle.SINGLE, size: 1, color: C.line },
    },
  });
}

function callout(titleText, bodyText, color = C.blue, fill = C.paleBlue) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    rows: [
      new TableRow({
        cantSplit: true,
        children: [
          cell(
            [
              paragraph([run(titleText, { bold: true, color, size: 22 })], {
                after: 45,
                keepNext: true,
              }),
              paragraph(bodyText, { after: 15, line: 310 }),
            ],
            CONTENT_WIDTH,
            { shading: fill, margin: 170 }
          ),
        ],
      }),
    ],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 8, color },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      left: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      right: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      insideHorizontal: { style: BorderStyle.NONE, size: 0, color: C.white },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: C.white },
    },
  });
}

function metricCards(items) {
  const widths = items.map(() => Math.floor(CONTENT_WIDTH / items.length));
  widths[widths.length - 1] += CONTENT_WIDTH - widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        cantSplit: true,
        children: items.map((item, index) =>
          cell(
            [
              paragraph([run(item.value, { bold: true, color: item.color || C.blue, size: 30 })], {
                alignment: AlignmentType.CENTER,
                after: 40,
              }),
              paragraph([run(item.label, { color: C.muted, size: 17 })], {
                alignment: AlignmentType.CENTER,
                after: 0,
              }),
            ],
            widths[index],
            { shading: item.fill || C.pale, margin: 160 }
          )
        ),
      }),
    ],
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      left: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      right: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      insideVertical: { style: BorderStyle.SINGLE, size: 1, color: C.white },
      insideHorizontal: { style: BorderStyle.NONE, size: 0, color: C.white },
    },
  });
}

function imageBlock(filename, width, height, caption) {
  const children = [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [
        new ImageRun({
          data: fs.readFileSync(path.join(ASSETS, filename)),
          type: "png",
          transformation: { width, height },
        }),
      ],
    }),
  ];
  if (caption) {
    children.push(
      paragraph([run(caption, { size: 17, color: C.muted, italics: true })], {
        alignment: AlignmentType.CENTER,
        after: 120,
      })
    );
  }
  return children;
}

function cover(titleText, subtitle, version, note) {
  const subtitleParagraphs = String(subtitle)
    .split("\\n")
    .map((line, index, lines) =>
      paragraph([run(line, { color: "DDE5EA", size: 26 })], {
        after: index === lines.length - 1 ? 500 : 50,
        line: 360,
      })
    );
  return [
    paragraph("", { after: 350 }),
    new Table({
      width: { size: CONTENT_WIDTH, type: WidthType.DXA },
      columnWidths: [CONTENT_WIDTH],
      rows: [
        new TableRow({
          cantSplit: true,
          children: [
            cell(
              [
                paragraph([run("AU VOLATILITY RESEARCH", { color: C.gold, size: 20, bold: true })], {
                  after: 200,
                }),
                paragraph([run(titleText, { color: C.white, size: 54, bold: true })], {
                  after: 170,
                  line: 420,
                }),
                ...subtitleParagraphs,
                paragraph([run(version, { color: C.gold, size: 22, bold: true })], {
                  after: 70,
                }),
                paragraph([run("2026-08-31｜Asia/Shanghai", { color: "C9D2D9", size: 19 })], {
                  after: 0,
                }),
              ],
              CONTENT_WIDTH,
              { shading: C.navy, margin: 480, verticalAlign: VerticalAlign.CENTER }
            ),
          ],
        }),
      ],
      borders: {
        top: { style: BorderStyle.NONE, size: 0, color: C.navy },
        bottom: { style: BorderStyle.NONE, size: 0, color: C.navy },
        left: { style: BorderStyle.NONE, size: 0, color: C.navy },
        right: { style: BorderStyle.NONE, size: 0, color: C.navy },
        insideHorizontal: { style: BorderStyle.NONE, size: 0, color: C.navy },
        insideVertical: { style: BorderStyle.NONE, size: 0, color: C.navy },
      },
    }),
    paragraph("", { after: 260 }),
    callout("文件性质", note, C.gold, C.paleGold),
    pageBreak(),
  ];
}

function docSettings(shortTitle) {
  return {
    styles: {
      default: {
        document: {
          run: { font: DOC_FONT, size: 21, color: C.ink },
          paragraph: { spacing: { line: 330, after: 110 } },
        },
        heading1: {
          run: { font: DOC_FONT, size: 34, bold: true, color: C.navy },
          paragraph: { spacing: { before: 280, after: 150 }, outlineLevel: 0 },
        },
        heading2: {
          run: { font: DOC_FONT, size: 28, bold: true, color: C.blue },
          paragraph: { spacing: { before: 190, after: 100 }, outlineLevel: 1 },
        },
        heading3: {
          run: { font: DOC_FONT, size: 24, bold: true, color: C.ink },
          paragraph: { spacing: { before: 150, after: 80 }, outlineLevel: 2 },
        },
      },
    },
    numbering: {
      config: [
        {
          reference: "bullets",
          levels: [
            {
              level: 0,
              format: LevelFormat.BULLET,
              text: "•",
              alignment: AlignmentType.LEFT,
              style: { paragraph: { indent: { left: 500, hanging: 260 } } },
            },
            {
              level: 1,
              format: LevelFormat.BULLET,
              text: "–",
              alignment: AlignmentType.LEFT,
              style: { paragraph: { indent: { left: 900, hanging: 260 } } },
            },
          ],
        },
        {
          reference: "numbers",
          levels: [
            {
              level: 0,
              format: LevelFormat.DECIMAL,
              text: "%1.",
              alignment: AlignmentType.LEFT,
              style: { paragraph: { indent: { left: 520, hanging: 300 } } },
            },
          ],
        },
      ],
    },
    sections: [
      {
        properties: {
          page: {
            size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
            margin: { top: 850, right: 850, bottom: 850, left: 850 },
          },
        },
        headers: {
          default: new Header({
            children: [
              paragraph([run(shortTitle, { size: 16, color: C.muted })], {
                alignment: AlignmentType.RIGHT,
                after: 0,
                border: {
                  bottom: { style: BorderStyle.SINGLE, size: 4, color: C.line },
                },
              }),
            ],
          }),
        },
        footers: {
          default: new Footer({
            children: [
              new Paragraph({
                alignment: AlignmentType.CENTER,
                children: [
                  run("研究用途｜不构成投资建议    ", { size: 16, color: C.muted }),
                  run("第 ", { size: 16, color: C.muted }),
                  new TextRun({ children: [PageNumber.CURRENT], size: 16, color: C.muted }),
                  run(" 页", { size: 16, color: C.muted }),
                ],
              }),
            ],
          }),
        },
        children: [],
      },
    ],
  };
}

const finalTrades = [
  ["2024-05-21", "5日", "au2406 / K576", "5", "8.83%", "-6,640", "-1.16%"],
  ["2024-10-29", "20日", "au2412 / K624", "5", "14.17%", "44,060", "7.05%"],
  ["2025-04-24", "20日", "au2506 / K784", "3", "4.03%", "109,870", "23.29%"],
  ["2025-05-30", "40日", "au2508 / K768", "1", "7.50%", "31,270", "20.21%"],
  ["2025-09-18", "5日", "au2510 / K832", "5", "4.19%", "-28,290", "-3.39%"],
  ["2025-10-28", "20日", "au2512 / K936", "5", "12.01%", "108,990", "11.76%"],
  ["2025-12-26", "20日", "au2602 / K1008", "2", "18.80%", "110", "0.03%"],
  ["2026-02-09", "5日", "au2603 / K1088", "1", "13.73%", "13,180", "5.99%"],
  ["2026-02-26", "20日", "au2604 / K1152", "4", "8.16%", "97,890", "10.65%"],
  ["2026-03-25", "40日", "au2606 / K976", "2", "14.26%", "130,810", "33.47%"],
  ["2026-04-20", "5日", "au2605 / K1056", "4", "18.87%", "2,560", "0.30%"],
  ["2026-07-21", "5日", "au2608 / K872", "5", "20.45%", "5,950", "0.68%"],
];

function buildStrategyManual() {
  const s = docSettings("沪金期权最终策略说明书｜10%风险预算＋每两日对冲");
  const c = s.sections[0].children;
  c.push(
    ...cover(
      "沪金期权最终策略说明书",
      "窄盘口≤25% · 10%风险预算 · 每2个交易日Delta对冲\n程序运行、成交口径、风险控制与移交手册",
      "冻结版本 final_narrow_rb10_hedge2d_20260830",
      "这是可复现的研究级策略包使用说明。程序可以重跑冻结历史回测，也可以在保持输入表结构不变时接入更新后的上游预测与盘口数据；它不是自动连接交易柜台的实盘下单系统。"
    ),
    heading("文档控制与使用边界", 1),
    table(
      ["项目", "内容"],
      [
        ["策略名称", "沪金ATM短跨式波动率策略：窄盘口≤25%、10%风险预算、每2日Delta对冲"],
        ["冻结版本", "final_narrow_rb10_hedge2d_20260830"],
        ["母版本", "strategy_v8_narrow_risk_hedge_grid_20260830"],
        ["初始研究资金", "人民币10,000,000元；仓位不复利"],
        ["评估窗口", "2021-07-01至2026-07-27，共5.073个日历年"],
        ["实际交易信号期", "2024-05-21至2026-07-21；12笔已到期交易"],
        ["可使用范围", "历史复现、滚动研究、纸面交易、执行审计、程序移交"],
        ["不包含", "实盘订单路由、经纪商风控、L2-L5期权逐档撮合、收益保证"],
      ],
      [2400, 7700]
    ),
    paragraph(""),
    callout(
      "最终选择",
      "用户最终选择10%风险预算＋每两个交易日对冲。该组合在18组网格中取得最高Sharpe 1.084936，净利润509,760元，仅比最高利润的12%＋每2日组合少3,540元，同时名义风险预算更低、资本效率更高。",
      C.gold,
      C.paleGold
    ),
    heading("阅读导航", 2),
    bullet("第1—3章：理解策略目标、模型信号以及为什么要双模型确认。"),
    bullet("第4—7章：理解入场、仓位、真实盘口执行和风险约束。"),
    bullet("第8—10章：理解回测口径、结果和10%／每2日参数的选择依据。"),
    bullet("第11—13章：照着运行程序、验证结果、替换未来数据并排查问题。"),
    bullet("附录：公式、数据文件、逐笔订单字段和冻结参数速查。"),
    pageBreak(),
    heading("一页执行摘要", 1),
    metricCards([
      { value: "¥509,760", label: "冻结样本净利润", color: C.green, fill: C.paleGreen },
      { value: "1.085", label: "年化Sharpe", color: C.blue, fill: C.paleBlue },
      { value: "-0.91%", label: "最大回撤", color: C.red, fill: "F7EAEA" },
      { value: "83.3%", label: "交易胜率", color: C.gold, fill: C.paleGold },
    ]),
    paragraph(""),
    paragraph(
      "策略预测的不是金价方向，而是未来5、20、40个交易日的实现方差。主模型和无IV稳健模型都认为市场隐含波动率在计入历史波动率风险溢价后仍明显高估未来实际波动，且不存在显著跳跃否决时，策略卖出真实挂牌的近平值认购与认沽，以期获得隐含波动率回落、时间价值衰减和实现波动低于定价的收益。"
    ),
    paragraph(
      "交易不是按中间价或日线理想价模拟。期权开仓按同步真实买一成交；初始期货对冲、持仓期再对冲及到期平仓全部按天勤历史真实一档盘口回放：买入只能吃卖一，卖出只能打买一，并且数量不得超过当时一档可见量。"
    ),
    callout(
      "必须先知道的风险",
      "最终结构仍是卖出未加保护翼的ATM跨式，理论尾部损失不封顶；回测最大回撤不是风险上限。程序包应先用于复现和前瞻纸面交易。实盘前必须补齐保护翼或组合级尾部限制、真实客户保证金、期权多档盘口与未成交处理。",
      C.red,
      "F7EAEA"
    ),
    ...imageBlock("final_equity_drawdown.png", 650, 379, "图1  最终策略冻结样本的累计权益与日度回撤。2021—2023无实际交易。"),
    heading("1. 策略定位与收益来源", 1, { pageBreakBefore: true }),
    heading("1.1 交易的是什么", 2),
    paragraph(
      "这是一套波动率相对价值策略。它不押注黄金上涨或下跌，而是判断期权市场收取的“保险费”是否高于未来实际发生的波动。核心比较对象不是裸的Forecast RV与IV，而是经过期限匹配和历史风险溢价校准后的公允波动率与当期ATM跨式隐含波动率。"
    ),
    table(
      ["概念", "专业定义", "直观解释"],
      [
        ["Forecast RV", "模型预测的未来5/20/40日实现方差或其年化波动率", "未来金价可能有多折腾"],
        ["Market IV", "由真实ATM跨式价格通过Black-76反解的到期隐含波动率", "市场当前卖的保险有多贵"],
        ["VRP", "隐含方差与未来实现方差之间的历史风险补偿", "卖保险本来就应收的一部分风险费"],
        ["Volatility Edge", "公允波动率减市场IV", "模型认为保险是便宜还是昂贵"],
      ],
      [1800, 4500, 3800]
    ),
    heading("1.2 收益构成", 2),
    bullet("期权部分：卖出跨式后，若到期内在价值低于收取的权利金，产生正的期权盈亏。"),
    bullet("Delta对冲部分：用黄金期货降低方向暴露；对冲本身可能盈利，也可能因追涨杀跌和价差产生损失。"),
    bullet("交易成本：显式手续费直接扣除；真实Bid/Ask跨价已经进入实际成交价格。"),
    bullet("风险来源：短Gamma、波动率跳升、夜盘跳空、流动性枯竭、保证金上升和模型状态变化。"),
    heading("1.3 当前策略只做空波动率", 2),
    paragraph(
      "研究阶段的多头波动率信号未能稳定转化为正的成本后收益，而短波动率经过模型择时后显著优于Persistence和无脑卖出。因此最终策略只保留主模型与稳健模型一致的空头信号。这里的“只做空”是样本证据结论，不是永恒市场规律。"
    ),
    heading("2. 预测模型与双模型确认", 1),
    heading("2.1 主模型", 2),
    paragraph(
      "主模型为HAR-MSE-log + Macro + GVZ + SLV IV + US EPU。HAR骨架包含log RV的1日、5日和22日尺度；宏观块包含COMEX非重叠波动、广义美元指数变化、美国10年实际利率变化和USD/CNY变化；期权前瞻块包含GVZ及SLV约30日隐含方差的水平与变化；US EPU用于刻画政策不确定性。5、20、40日共用同一结构，但分别估计系数。"
    ),
    heading("2.2 无IV稳健模型", 2),
    paragraph(
      "稳健模型为HAR-QLIKE + Macro + US EPU。它不使用GVZ和SLV IV，训练目标为QLIKE等价的Gamma deviance。该模型的作用不是取代主模型，而是防止含IV模型只是复述期权市场本身：只有不看IV的模型也认为未来实际波动偏低时，卖方信号才成立。"
    ),
    heading("2.3 为什么加入IV不会形成完全自证", 2),
    bullet("预测目标是未来AU实现方差，不是期权价格或未来IV。"),
    bullet("主模型中的GVZ/SLV IV是跨市场前瞻信息，不是当前AU ATM IV本身。"),
    bullet("最终交易要求无IV稳健模型同方向、同样超过1.5个波动率点门槛。"),
    bullet("市场IV还要经过独立的历史VRP校准后才与模型比较。"),
    callout(
      "双模型原则",
      "主模型提供信息效率，稳健模型提供独立性。两者一致才交易；分歧时宁可不做。",
      C.green,
      C.paleGreen
    ),
    ...imageBlock("model_qlike_comparison.png", 650, 375, "图2  主模型与无IV稳健模型相对Persistence的全样本QLIKE。100%代表Persistence；越低越好。"),
    heading("3. 信号形成：从预测到可交易机会", 1, { pageBreakBefore: true }),
    heading("3.1 期限匹配与VRP校准", 2),
    paragraph([
      run("年化公允方差 = 252 × 预测日方差 + 历史VRP估计；", { bold: true }),
      run("公允波动率为该方差的平方根。VRP只使用在信号日之前已经到期的同期限历史机会，并对10%—90%分位之外的极端值缩尾后取中位数。"),
    ]),
    paragraph([
      run("Edge = 公允波动率 − 当期ATM IV。", { bold: true }),
      run("最终做空要求主模型Edge≤−1.5 vol points且稳健模型Edge≤−1.5 vol points。"),
    ]),
    heading("3.2 信号硬条件", 2),
    table(
      ["层级", "冻结规则", "失败处理"],
      [
        ["模型方向", "主模型与无IV稳健模型都给出负Edge", "不交易"],
        ["最小优势", "两套模型的绝对负Edge均不少于1.5 vol points", "不交易"],
        ["跳跃否决", "信号日不得检测到显著Jump", "禁止新增短波动"],
        ["期限", "5、20、40交易日机会均可", "其他期限不进入"],
        ["数据成熟", "训练目标到信号日前已经成熟；信息时间戳不晚于15:05截点", "不训练、不预测"],
        ["基础流动性", "ATM偏离≤1.5%、两腿有成交、组合OI≥20、权利金≥6 ticks", "不交易"],
        ["真实窄盘口", "入场跨式相对价差≤25%", "不交易"],
      ],
      [1650, 5850, 2600]
    ),
    heading("3.3 相对价差定义", 2),
    paragraph(
      "相对价差 =（买入跨式成本 − 卖出跨式收入）÷ 跨式中间价。卖出跨式收入使用Call买一＋Put买一；买入跨式成本使用Call卖一＋Put卖一。冻结上限为25%。该过滤器是最终策略最重要的执行质量门槛之一。"
    ),
    heading("4. 合约与入场执行", 1),
    heading("4.1 合约选择", 2),
    bullet("只使用当时真实挂牌、与对应黄金期货合约相匹配的上期所黄金期权。"),
    bullet("选择同一到期、同一行权价、最接近ATM的认购与认沽组成跨式。"),
    bullet("预测期限5/20/40日对应从信号日至期权到期的交易日距离，而不是假想常数期限合约。"),
    bullet("信号日为t，真实入场日在下一交易日t+1；避免用信号日收盘价假装可成交。"),
    heading("4.2 入场真实盘口", 2),
    table(
      ["腿", "动作", "成交价", "时间同步/容量"],
      [
        ["ATM Call", "卖出", "真实买一 bid1", "与Put、期货最大时间差≤10秒；期权容量按最小显示量×5"],
        ["ATM Put", "卖出", "真实买一 bid1", "同上"],
        ["黄金期货", "建立初始Delta对冲", "买入=卖一ask1；卖出=买一bid1", "数量不得超过当时成交侧一档可见量"],
      ],
      [1700, 1500, 2600, 4300]
    ),
    callout(
      "5×期权深度的真实含义",
      "价格是真实bid1，但容量允许使用当时期权一档显示量的5倍。这是基于盘口回补能力的研究假设，不是L2-L5逐档历史扫单。期货侧则严格限定在真实一档可见数量内。",
      C.gold,
      C.paleGold
    ),
    heading("5. 仓位与风险预算", 1),
    heading("5.1 单笔风险资本", 2),
    paragraph([
      run("每手风险资本 = 20% × 入场期货中间价 × 1,000克/手。", { bold: true }),
      run("风险预算手数 = floor（10,000,000 × 10% ÷ 每手风险资本）。仓位不复利，因此历史收益不会自动扩大下一笔规模。"),
    ]),
    heading("5.2 实际成交手数", 2),
    paragraph(
      "最终手数取以下约束的最小值：风险预算手数、期权5×显示深度容量、30%组合同时风险上限允许手数、以及完整期货对冲路径在真实一档盘口下可执行的手数。若后续或到期对冲无法完整执行，程序逐级减少该笔期权手数，直到整条路径可成交。"
    ),
    table(
      ["风险约束", "冻结值", "作用"],
      [
        ["单笔风险预算", "初始资金10%", "控制理论可下单手数"],
        ["短跨风险资本系数", "标的名义价值20%", "近似卖方保证金与压力资金"],
        ["组合同时风险上限", "初始资金30%", "限制重叠到期仓位"],
        ["实际历史最大同时风险", "13.10%", "当前样本未触及30%上限"],
        ["Delta最小调仓单位", "变化至少1手", "避免0手或无意义调整"],
      ],
      [2600, 2300, 5200]
    ),
    heading("6. 每两日Delta对冲", 1),
    heading("6.1 调仓时间", 2),
    paragraph(
      "入场时先按同步期货盘口建立初始Delta对冲。之后以交易日计数，每2个交易日在15:00目标时刻前最后一条有效期货盘口上重算跨式Delta并调整；到期日无论是否轮到计划调仓，均必须把期货对冲完整平仓。"
    ),
    heading("6.2 成交纪律", 2),
    bullet("期货买入成交价必须等于当时ask1。"),
    bullet("期货卖出成交价必须等于当时bid1。"),
    bullet("成交量不得超过对应一档可见数量；可部分成交。"),
    bullet("持仓期浮盈亏使用立即可平仓一侧估值：多头按bid1、空头按ask1。"),
    bullet("禁止使用日线收盘价、mid加一跳或人为滑点替代成交。"),
    heading("6.3 为什么不是每日或每三日", 2),
    paragraph(
      "在10%风险预算下，每两日对冲净利润509,760元、Sharpe 1.084936；每日对冲净利润464,830元、Sharpe 1.060745；每三日对冲净利润306,580元、Sharpe 0.598640。两日规则在当前样本中减少不必要的盘口跨越，又没有像三日规则那样积累过多方向暴露。该结论仍需前瞻样本确认。"
    ),
    heading("7. 到期、估值与盈亏", 1),
    bullet("期权持有到到期，按真实到期期货top-book中间价计算跨式内在价值 |F−K|；这是结算估值，不是期货成交。"),
    bullet("到期期货对冲按真实bid1/ask1平仓，且必须完整可成交；否则回溯降低初始期权手数。"),
    bullet("中间期权日度标记使用当日期权收盘价；若缺失则用上次有效IV的Black-76价格。这会影响日度Sharpe和最大回撤路径，但不改变最终真实入场及到期实现利润。"),
    bullet("显式费用：期权每腿每手5元、行权/履约每腿每手5元、期货每手每边5元。"),
    heading("8. 回测设计与真实性审计", 1, { pageBreakBefore: true }),
    heading("8.1 样本", 2),
    paragraph(
      "评估窗口为2021-07-01至2026-07-27，共1,229个交易日、5.073个日历年。模型预测从2021年已可用，但最终做空条件直到2024-05-21才首次满足，因此实际有仓位的证据只有约两年两个月、12个到期簇。"
    ),
    heading("8.2 审计结果", 2),
    metricCards([
      { value: "104", label: "订单/结算行", color: C.blue },
      { value: "56", label: "期货真实盘口订单", color: C.green },
      { value: "0", label: "无效期货订单", color: C.gold },
      { value: "250/250", label: "收盘盘口覆盖", color: C.blue },
    ]),
    paragraph(""),
    bullet("250个期货合约×交易日收盘快照均有效；最大报价年龄不到1秒。"),
    bullet("全部56笔期货订单通过方向价格、时间戳、真实数据源和一档可见量检查。"),
    bullet("订单手续费与交易级显式成本逐笔对账，误差为0。"),
    bullet("最终单策略与V8中10%／每2日组合的利润、Sharpe、回撤、手数和对冲周转一致。"),
    heading("9. 专业回测结果", 1),
    metricCards([
      { value: "5.10%", label: "总收益率", color: C.green, fill: C.paleGreen },
      { value: "1.02%", label: "年化收益率", color: C.blue, fill: C.paleBlue },
      { value: "0.94%", label: "年化波动率", color: C.blue, fill: C.paleBlue },
      { value: "15.59", label: "Profit Factor", color: C.gold, fill: C.paleGold },
    ]),
    paragraph(""),
    table(
      ["指标", "结果", "解释"],
      [
        ["净利润", "509,760元", "10,000,000元固定本金口径"],
        ["Sharpe", "1.084936", "基于1,229个交易日的日度权益变化"],
        ["Sortino", "0.527835", "只用下行波动作为风险分母"],
        ["最大回撤", "-0.9092%", "日度标记权益的历史峰谷损失"],
        ["胜率", "83.33%", "12笔中10笔盈利"],
        ["平均/中位单笔", "42,480 / 22,225元", "利润分布右偏，少数大盈利重要"],
        ["显式手续费", "1,200元", "不含已嵌入真实价格的买卖价差"],
        ["观察到的总摩擦", "117,990元", "手续费＋期权/期货相对中间价跨价；不要再次从净利润扣除"],
        ["Bootstrap 95%区间", "173,328—872,638元", "按交易/到期簇重采样；小样本下仅作不确定性参考"],
      ],
      [2500, 2600, 5000]
    ),
    ...imageBlock("final_trade_pnl.png", 650, 343, "图3  12笔最终策略交易的实现净盈亏。"),
    heading("9.1 年度分布", 2),
    table(
      ["年份", "净利润", "交易数", "胜率"],
      [
        ["2021", "0", "0", "—"],
        ["2022", "0", "0", "—"],
        ["2023", "0", "0", "—"],
        ["2024", "37,420", "2", "50%"],
        ["2025", "207,615", "5", "80%"],
        ["2026至7月27日", "264,725", "5", "100%"],
      ],
      [1800, 2500, 1800, 2000]
    ),
    heading("9.2 全部12笔交易", 2, { pageBreakBefore: true }),
    table(
      ["入场日", "期限", "合约/行权价", "手数", "相对价差", "净盈亏", "风险资本收益"],
      finalTrades,
      [1250, 750, 2200, 600, 1200, 1300, 1500],
      { fontSize: 15, alignments: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.RIGHT] }
    ),
    heading("10. 为什么最终选10%＋每两日", 1, { pageBreakBefore: true }),
    ...imageBlock("grid_sharpe_heatmap.png", 650, 425, "图4  6档风险预算×3档Delta对冲频率的完整Sharpe网格。"),
    table(
      ["候选", "净利润", "Sharpe", "最大回撤", "判断"],
      [
        ["原始7.5%＋每日", "402,270", "1.0149", "-1.07%", "冻结母策略"],
        ["10%＋每日", "464,830", "1.0607", "-1.08%", "提高仓位但对冲较频繁"],
        ["10%＋每2日", "509,760", "1.0849", "-0.91%", "最终选择；全场最高Sharpe"],
        ["12%＋每2日", "513,300", "1.0845", "-0.91%", "利润冠军，但仅多3,540元"],
        ["15%/18%＋每2日", "513,300", "1.0845", "-0.91%", "受期权5×和期货一档容量封顶"],
      ],
      [2300, 1700, 1500, 1700, 2900]
    ),
    paragraph(""),
    callout(
      "选择逻辑",
      "10%＋每2日以少0.69%的净利润换取全场最高Sharpe、较低名义风险预算和更好的未来容量纪律。12%—18%在历史上结果相同，是流动性封顶，而不是高预算没有风险。",
      C.green,
      C.paleGreen
    ),
    heading("11. 程序包使用方法", 1),
    heading("11.1 包内结构", 2),
    table(
      ["路径", "用途"],
      [
        ["final_strategy_10pct_2d_config.yaml", "唯一冻结策略配置"],
        ["scripts/27_run_final_strategy_10pct_2d.py", "运行单策略真实盘口回测"],
        ["scripts/28_verify_final_strategy_package.py", "校验文件哈希、参考指标和真实成交"],
        ["src/au_rv/", "模型、特征、Black-76、策略和真实盘口执行源码"],
        ["data/", "冻结回测所需的最小输入快照"],
        ["reference_results/", "交付时的参考指标、交易、订单和审计文件"],
        ["docs/", "本说明书与完整研究报告"],
        ["package_manifest.json", "包内文件SHA-256清单；用于确认未被修改"],
      ],
      [3900, 6200]
    ),
    heading("11.2 首次运行", 2),
    callout(
      "命令",
      "1）python3 -m venv .venv；2）.venv/bin/pip install -r requirements-lock.txt；3）.venv/bin/python scripts/27_run_final_strategy_10pct_2d.py --verify-reference；4）.venv/bin/python scripts/28_verify_final_strategy_package.py --verify-reference。",
      C.blue,
      C.paleBlue
    ),
    paragraph(
      "若使用Windows，请把“.venv/bin/python”替换为“.venv\\Scripts\\python.exe”。首次冻结复现不需要API密钥，因为包中已包含最小历史输入；不要把主项目的真实.env复制进对外分发包。"
    ),
    heading("11.3 输出文件", 2),
    bullet("metrics.csv：策略级收益、风险和交易统计。"),
    bullet("trades.csv：每笔期权组合的入场、仓位、成本、对冲与最终净盈亏。"),
    bullet("orders.csv：全部期权开仓、期货对冲、期权到期结算记录。"),
    bullet("futures_execution_audit.csv：每笔期货订单的bid/ask方向、可见量和数据源审计。"),
    bullet("daily.csv / annual.csv / monthly.csv：日度权益和分期绩效。"),
    bullet("freeze_manifest.json：本次运行的配置、源码、数据和结果哈希。"),
    heading("11.4 接入未来数据", 2),
    paragraph(
      "未来重跑时，上游预测管线必须生成与bundled opportunities.parquet相同字段结构的机会表，并更新期权日线、期货日线、同步入场tick和15:00期货top-book tick。保持最终策略参数不变时仍可沿用版本逻辑；如果改变风险预算、对冲频率、价差阈值或信号模型，必须新建版本号和输出目录。"
    ),
    table(
      ["必须更新的输入", "最低要求"],
      [
        ["opportunities.parquet", "包含主/稳健预测、VRP校准、信号、合约、到期和流动性字段"],
        ["期权入场tick", "Call、Put、期货同步有效bid1/ask1及挂单量，最大时间差10秒"],
        ["期货持仓期tick", "每个所需交易日15:00前有效top-book，含数量和时间戳"],
        ["期权/期货日线", "用于期权日度标记、期限路径和到期校验"],
      ],
      [3300, 6800]
    ),
    heading("11.5 与独立预测模型包的关系", 2),
    paragraph(
      "本次另行交付final_dual_har_x_20260831预测模型包。模型包保存主/稳健两套HAR-X在5、20、40日上的六个冻结模型文件、15项输入特征模板、命令行调用器、本地可视化页面和Walk-forward评价。预测模型包只输出未来实现波动率；策略包在其上继续完成VRP校准、与AU期权IV比较、信号过滤、仓位、真实Bid/Ask成交与Delta对冲。两者分开可以独立升级和审计。"
    ),
    callout(
      "接入原则",
      "模型输出不是交易信号。只有主模型和无IV稳健模型同向，并且经过因果VRP校准后仍显示期权IV足够昂贵，才进入最终策略的盘口与风险过滤。",
      C.blue,
      C.paleBlue
    ),
    heading("12. 运行前后检查清单", 1),
    heading("12.1 运行前", 2),
    numbered("确认系统日期、时区和回测evaluation_end，不允许未来未完成交易混入。"),
    numbered("确认配置仍通过冻结校验；不要直接改10%、2日或25%并沿用原版本号。"),
    numbered("确认数据文件完整，API密钥只放本地.env，包内没有真实凭据。"),
    numbered("确认最新模型输出的特征截止时间不晚于信号日15:05。"),
    numbered("确认所有已开机会都有覆盖到到期的真实期货top-book路径。"),
    heading("12.2 运行后", 2),
    numbered("运行verify脚本；invalid_futures_orders必须为0。"),
    numbered("检查买期货=ask1、卖期货=bid1、quantity≤对应一档量。"),
    numbered("检查order_reconciliation的最大误差≤1e-6。"),
    numbered("抽查新增交易的信号日、入场日、到期日、期权腿和对冲时间戳。"),
    numbered("把新结果标记为前瞻、滚动研究或历史重算，不能混淆。"),
    heading("13. 局限性与实盘前必做事项", 1),
    table(
      ["优先级", "问题", "影响", "建议"],
      [
        ["P0", "只有12笔交易", "Sharpe和胜率极不稳定", "冻结规则后积累至少20—30笔新交易"],
        ["P0", "短跨尾部损失不封顶", "跳空、涨停、IV暴涨可能远超历史回撤", "加入真实可成交保护翼或账户级压力上限"],
        ["P0", "期权5×深度是假设", "大单真实冲击可能被低估", "保存期权L2-L5、成交队列和回补速度"],
        ["P0", "真实客户保证金未知", "强平与资金占用可能偏差", "接入期货公司客户保证金与手续费账单"],
        ["P1", "日度期权标记非逐笔清算价", "影响Sharpe和回撤路径", "补齐每日期权收盘top-book并保守标记"],
        ["P1", "18组参数同样本比较", "冠军存在多重检验偏差", "外层walk-forward或纯前瞻验证"],
        ["P1", "尚无自动下单与故障处置", "程序不能直接承担实盘执行", "增加风控网关、限价、撤单、断线和审计"],
      ],
      [1000, 2700, 3000, 3400],
      { fontSize: 16 }
    ),
    pageBreak(),
    heading("附录A：冻结参数速查", 1),
    table(
      ["参数", "冻结值"],
      [
        ["策略名", "final_narrow_rb10_hedge2d"],
        ["方向", "只做空波动率"],
        ["预测期限", "5、20、40交易日"],
        ["最小双模型Edge", "1.5 vol points"],
        ["最大跨式相对价差", "25%"],
        ["单笔风险预算", "初始资金10%"],
        ["组合风险上限", "初始资金30%"],
        ["期权容量", "最小bid1显示量×5"],
        ["对冲频率", "每2个交易日"],
        ["Delta调仓门槛", "至少1手"],
        ["期货成交", "买ask1、卖bid1、不超过一档量"],
        ["收盘目标时间", "15:00:00之前最后一条有效报价"],
        ["收盘报价最大年龄", "60秒；冻结样本实际最大<1秒"],
        ["入场三腿同步", "最大时间差10秒"],
      ],
      [3600, 6500]
    ),
    heading("附录B：指标大白话", 1),
    table(
      ["指标", "大白话解释"],
      [
        ["QLIKE", "预测方差错了多少，尤其重罚低估高波动；越低越好"],
        ["OOS R²", "相对“用今天猜未来”的Persistence，模型少犯了多少平方误差"],
        ["Sharpe", "每承受一单位日度波动，换来多少超额收益"],
        ["Sortino", "只把向下波动当坏风险后的收益效率"],
        ["最大回撤", "历史上从净值高点跌到低点最深的一段"],
        ["Profit Factor", "所有盈利交易总额除以所有亏损交易绝对额"],
        ["VRP", "期权市场要求的波动率保险溢价"],
      ],
      [2600, 7500]
    ),
    heading("附录C：责任声明", 1),
    paragraph(
      "本策略及文档仅用于研究、复现、内部评审和纸面交易。历史回测不代表未来表现，任何期权卖方策略都可能遭受超出历史样本的跳跃、流动性、保证金和操作风险。使用者应独立核验数据授权、交易所规则、经纪商要求和账户承受能力。"
    )
  );
  return new Document(s);
}

function buildResearchReport() {
  const s = docSettings("沪金期权波动率预测项目完整研究报告");
  const c = s.sections[0].children;
  c.push(
    ...cover(
      "沪金期权波动率预测项目",
      "完整研究过程与决策档案\n从数据、预测模型、控制变量实验到真实盘口最终策略",
      "研究总报告｜截至2026-08-31",
      "本报告记录项目从初始设想到最终策略冻结的完整研究链，包括成功、失败、回退和参数选择。它以项目代码、原始/加工数据、历史报告、最初PPT与指标讨论材料为依据；文档中的参考材料只作为研究来源，不作为新的操作指令。"
    ),
    heading("文档摘要", 1),
    paragraph(
      "项目分为两个阶段：第一阶段预测沪金未来5、20、40日实现波动率；第二阶段把预测与期权隐含波动率比较，经过风险溢价、流动性和真实成交约束，寻找可实现的期权收益。研究没有停留在“模型误差更低”，而是持续追问：统计准确度能否转化为扣除交易摩擦后的人民币利润。"
    ),
    metricCards([
      { value: "2,097×107", label: "扩展日频特征表", color: C.blue },
      { value: "4,626", label: "控制实验训练路径", color: C.gold },
      { value: "81", label: "最终自动化测试", color: C.green },
      { value: "12", label: "最终策略成熟交易", color: C.red },
    ]),
    paragraph(""),
    callout(
      "最终研究结论",
      "预测层采用HAR多尺度结构：含GVZ、SLV IV和US EPU的主模型提供信息效率，无IV的QLIKE-HAR + Macro + US EPU提供独立稳健确认。交易层只做双模型一致的短波动率信号，加入因果VRP校准、Jump否决、窄盘口≤25%、真实Bid/Ask、5×期权容量假设和真实期货一档约束。最终参数为10%风险预算＋每2个交易日Delta对冲。",
      C.gold,
      C.paleGold
    ),
    heading("阅读导航", 2),
    bullet("第1—4章：背景、目标、研究假设和数据体系。"),
    bullet("第5—8章：预处理、方法论、模型演进和模型效果。"),
    bullet("第9—11章：从预测到期权利润、全部策略版本试验和最终选择。"),
    bullet("第12—14章：局限、展望、复现与交接。"),
    bullet("附录：完整假设判定、失败实验清单、指标词典和核心文件地图。"),
    ...imageBlock("research_timeline.png", 680, 272, "图1  项目从问题定义到最终冻结的研究演进。"),
    pageBreak(),
    heading("1. 项目背景", 1),
    paragraph(
      "黄金期权的市场价格同时包含未来波动预期、风险厌恶、尾部保险需求、流动性和供需。单纯预测黄金涨跌方向很难直接解释期权贵不贵；而预测未来实现方差，再与同期限隐含波动率比较，可以更直接地识别波动率相对定价。"
    ),
    paragraph(
      "最初汇报PPT提出了一套三层架构：统计预测引擎处理正常状态，事件预警引擎识别状态切换，交易决策引擎把Forecast RV、Market IV、DTE、流动性和风险限制合成动作。项目当前完成了高质量的统计预测、波动率交易基线和真实盘口回测；实时新闻事件重放、自动下单和实盘风控网关仍属于下一阶段。"
    ),
    heading("1.1 为什么选择沪金", 2),
    bullet("黄金受美元、实际利率、地缘风险和全球风险偏好共同驱动，波动率具有跨市场信息基础。"),
    bullet("上期所黄金期货与期权有真实挂牌、合约到期和夜盘结构，适合检验模型能否落地。"),
    bullet("黄金波动率并不像股票那样固定与价格负相关，方向中性Delta对冲更符合研究目标。"),
    heading("1.2 项目要解决的核心矛盾", 2),
    paragraph(
      "预测模型可以在统计上更准确，却不一定让期权策略赚钱。期权利润还受IV风险溢价、到期匹配、Gamma/Theta路径、Bid/Ask、盘口容量、Delta对冲、保证金和尾部跳跃影响。因此项目从一开始就把“预测评价”和“交易评价”分开，并设置Persistence、纯IV、HAR/HARQ、GARCH和无脑卖波动率等对照。"
    ),
    heading("2. 项目目标", 1),
    heading("2.1 第一阶段：波动率预测", 2),
    numbered("从沪金5分钟行情构造因果、可审计的日度实现方差与多尺度特征。"),
    numbered("分别预测未来5、20、40个交易日的日均实现方差，并统一输出年化波动率。"),
    numbered("使用严格Walk-forward样本外评价，比较Persistence、IV、HAR、HARQ、GARCH及外生变量组合。"),
    numbered("形成可重复训练、可保存模型版本、可生成最新预测的工程管线。"),
    heading("2.2 第二阶段：期权策略", 2),
    numbered("将期限匹配的Forecast RV与真实期权ATM IV比较，并校准历史波动率风险溢价。"),
    numbered("只在模型优势、流动性、Jump状态和组合风险同时通过时交易。"),
    numbered("使用真实挂牌合约、真实Bid/Ask、挂单量和期货Delta对冲路径回测。"),
    numbered("以净利润、Sharpe、回撤、成本、容量和可复现性共同评价，而不是只看预测误差。"),
    heading("2.3 成功标准", 2),
    table(
      ["层级", "成功标准"],
      [
        ["数据", "时间戳、交易日、夜盘、合约选择和数据可得性不泄漏未来"],
        ["预测", "大部分期限在QLIKE/RMSE/OOS R²上稳定优于Persistence；误差可解释"],
        ["经济", "模型择时策略优于Persistence信号和无脑卖；成本后仍为正"],
        ["执行", "真实Bid/Ask和可见量可追溯；订单级成本对账"],
        ["风险", "承认尾部、保证金和小样本；有明确不上线条件"],
        ["工程", "配置、代码、数据与输出有版本和SHA-256清单，测试可重复"],
      ],
      [2200, 7900]
    ),
    heading("3. 研究假设", 1),
    table(
      ["编号", "研究假设", "最终证据"],
      [
        ["H1", "波动率具有多尺度持续性，HAR日/周/月结构有效", "支持；HAR家族在5/20日稳定强于Persistence"],
        ["H2", "黄金跨市场和宏观变量可提供增量信息", "部分支持；Macro、GVZ、US EPU稳定，GEPU/GPR平均贡献偏弱"],
        ["H3", "QLIKE比MSE-log更适合方差风险", "支持但非所有指标同升；v1.1 QLIKE明显下降，RMSE并非总改善"],
        ["H4", "HARQ能通过测量误差修正普遍提升预测", "只在部分长周期有小增益；5/20日不支持"],
        ["H5", "IV含前瞻信息，但直接当预测会自证或偏置", "支持；校准IV可有用，raw IV很差；最终采用含IV主模型＋无IV稳健模型"],
        ["H6", "GARCH可以统一替代HAR", "不支持；40日raw GARCH有价值，但GARCH-X没有统一优势"],
        ["H7", "更准确的RV预测能转化为期权收益", "支持但依赖VRP和执行；模型择时明显优于Persistence和无脑卖"],
        ["H8", "真实成交摩擦会显著降低理想利润", "强支持；真实期货盘口使多个策略利润下降14%—48%"],
        ["H9", "更保守一定更好", "不支持；v2过度过滤只剩2笔，保守组合在V7反而表现最差"],
      ],
      [700, 4700, 4700],
      { fontSize: 16 }
    ),
    heading("4. 数据与数据源", 1, { pageBreakBefore: true }),
    heading("4.1 核心数据源", 2),
    table(
      ["数据源", "数据与频率", "用途", "关键限制"],
      [
        ["天勤量化 TQSDK", "AU期货5分钟；期权/期货日线；期权与期货tick；交易日历", "RV、Jump、合约选择、期权IV、真实入场与Delta对冲", "会员权限；历史盘口主要为L1；夜盘归属需自行统一"],
        ["Databento GLBX.MDP3", "GC.v.0 1分钟", "COMEX与沪金非重叠时段波动", "控制下载成本；连续合约映射需按当时成交量"],
        ["Databento OPRA/ARCX", "SLV期权/标的日线", "自行反解约30日SLV IV", "不是AU IV；定义与价格数据需匹配"],
        ["FRED/ALFRED", "美元、实际利率、USD/CNY、GVZ、US EPU、GEPU", "宏观与期权前瞻块", "使用初次发布/保守可得时间，避免修订泄漏"],
        ["GPR作者官网", "日度地缘政治风险指数", "控制变量", "工作簿会修订；采用保守发布滞后"],
        ["BLS/Federal Reserve/FRED release", "CPI、NFP、FOMC日历", "未来已知事件计数", "只能用当时已经公布的日历"],
      ],
      [2000, 2700, 2700, 2700],
      { fontSize: 15 }
    ),
    heading("4.2 数据规模", 2),
    metricCards([
      { value: "1,388,874", label: "扩展AU 5分钟行", color: C.blue },
      { value: "3,710,819", label: "GC 1分钟行", color: C.gold },
      { value: "13,811", label: "FRED初次发布行", color: C.green },
      { value: "15,211", label: "GPR日度行", color: C.red },
    ]),
    paragraph(""),
    bullet("扩展日频特征：2,097行×107列，2018-01-02至2026-08-24。"),
    bullet("feature_valid_flag通过1,455行；更严格的完整控制实验共同有效行1,116。"),
    bullet("SLV约30日IV序列1,752行；不是直接下载一个IV字段，而是用期权和标的价格重建。"),
    bullet("真实期货Delta对冲：170,774条历史tick，250个所需收盘快照全部覆盖。"),
    heading("4.3 成本和凭据治理", 2),
    paragraph(
      "Databento下载前先估价，小区间验证后才执行批量请求，项目设置25美元上限；新增QLIKE/HARQ时复用已有原始数据，没有重复付费。API密钥仅保存在本地.env，正式交付包只保留变量名和.env.example，不包含任何凭据值。"
    ),
    heading("5. 数据预处理与防前视", 1),
    heading("5.1 时间口径", 2),
    bullet("统一保存原始时区、UTC和Asia/Shanghai时间。"),
    bullet("沪金夜盘按交易所交易日归属，不按自然日简单切分。"),
    bullet("每日模型信息截点为15:05；只保留该时刻前已知的数据。"),
    bullet("FRED、GPR和事件日历通过available_at向后as-of join，并设置数据陈旧上限。"),
    bullet("目标从t+1开始；训练只使用maturity_date≤当前预测日的历史目标。"),
    heading("5.2 AU实现方差与质量控制", 2),
    paragraph(
      "将有效5分钟对数收益平方和作为日度实现方差RV，并保留日、5日、22日log RV。每天根据点时成交和合约规则选择代表AU合约；要求足够的盘中bar、无异常时间间隔、完整交易日和明确的数据质量标志。"
    ),
    heading("5.3 Jump与HARQ", 2),
    paragraph(
      "使用Realized Variance与Bipower Variation分解连续和跳跃部分，并以显著性检验产生jump_significant。HARQ计算Realized Quarticity并构造log RV×sqrt(RQ)交互项，用于让短期持续性随高频RV测量误差变化。"
    ),
    heading("5.4 COMEX非重叠信息", 2),
    paragraph(
      "GC 1分钟数据先聚合到5分钟，只使用在沪金15:05信息截点前、且不与沪金本地交易时段重叠的COMEX收益，避免把同一价格冲击重复计入或使用未来美盘数据。"
    ),
    heading("5.5 目标变量", 2),
    paragraph(
      "5、20、40日目标为从下一交易日开始的未来日均实现方差；建模时同时保存方差和log方差，预测后按252个交易日年化并开平方得到Forecast Vol。直接多期限建模避免先预测1日再机械迭代累积误差。"
    ),
    callout(
      "防泄漏原则",
      "任何字段如果在预测日15:05以后才知道、需要未来完整样本才能计算，或对应目标尚未成熟，就不能进入该次训练和预测。available_at、maturity_date、purged time-series CV和月度滚动重训共同执行这一原则。",
      C.green,
      C.paleGreen
    ),
    heading("6. 方法论：为什么选这些模型", 1),
    heading("6.1 Ridge-HAR-X", 2),
    paragraph(
      "HAR以日、周、月三种市场参与者时间尺度解释波动聚集和长记忆；黄金高质量样本有限，宏观变量高度共线，因此使用L2正则化压缩不稳定系数。它在经济解释、有限样本、滚动稳定性和工程可维护性之间优于高自由度黑箱模型。"
    ),
    heading("6.2 QLIKE", 2),
    paragraph(
      "QLIKE是面向条件方差预测的稳健损失，对高波动时的低估惩罚更重。项目以带log link的GammaRegressor实现QLIKE等价的Gamma deviance，保证方差预测严格为正；正则强度使用带horizon gap的时间序列交叉验证选择。"
    ),
    heading("6.3 Walk-forward", 2),
    paragraph(
      "每个预测时点只使用当时已成熟历史，采用扩展训练窗和月度重训。模型评估只看样本外预测；超参数选择、模型选择和信号生成均保留日期与训练样本数。这样能最大限度模拟真实研究和上线顺序。"
    ),
    heading("6.4 控制变量实验", 2),
    paragraph(
      "HAR、HARQ和GARCH三种基座分别用MSE-log与QLIKE训练，对8个变量块的256个子集全组合；每个期限1,542个估计模型，另有原始基准，三期限共4,626条训练路径。评价分为截至2025的选择段、2026时间验证段和全样本描述段，并使用DM-HLN检验与FDR控制多重比较。"
    ),
    heading("6.5 为什么最终不使用GARCH作统一主模型", 2),
    paragraph(
      "原始GARCH(1,1)在40日全样本QLIKE上表现突出，说明其长期均值回归仍有独立价值；但5日明显弱、GARCH-X全组合没有获得跨期限统一优势。用户最终选择保留HAR主模型和稳健对照，不把GARCH纳入生产决策链。"
    ),
    heading("7. 模型演进与结论", 1),
    heading("7.1 从“50个指标”回到可控结构", 2),
    paragraph(
      "早期材料列出接近50个字段，引发是否需要大回归的质疑。研究随后明确区分：数据主键、目标构造、预测特征、执行字段、交易过滤、风险字段和模型输出。最终不是把全部字段塞进回归，而是用小型HAR骨架与分块控制实验筛选增量信息。"
    ),
    heading("7.2 v1.0：log-RV Ridge基线", 2),
    paragraph(
      "第一版建立HAR、HAR+Jump、HAR+Jump+Macro等Ridge模型，完成数据下载、特征表、Walk-forward和Persistence比较。它验证了多尺度RV与宏观信息的可用性，也暴露出MSE-log可能低估高波动风险的问题。"
    ),
    heading("7.3 v1.1：QLIKE与HARQ", 2),
    table(
      ["期限", "冠军模型", "QLIKE", "log-RV RMSE", "OOS R²", "方向准确率"],
      [
        ["5日", "QLIKE-HAR", "0.1843", "0.5443", "0.3788", "69.57%"],
        ["20日", "QLIKE-HAR + Jump + Macro", "0.2715", "0.6421", "0.4037", "72.23%"],
        ["40日", "QLIKE-HARQ + Jump + Macro", "0.3406", "0.6890", "0.4449", "74.46%"],
      ],
      [900, 3600, 1250, 1500, 1200, 1500],
      { fontSize: 16 }
    ),
    paragraph(
      "相对v1.0冠军，QLIKE下降10.65%/9.31%/5.75%，但5日和20日RMSE略升。这说明QLIKE改善的是方差风险口径，不应包装成所有指标全面提高。HARQ在5/20日无增益，仅在40日宏观模型上带来约0.17%的小改善。"
    ),
    heading("7.4 扩展控制实验", 2),
    bullet("US EPU是最稳定的增量块之一；Macro在20/40日稳定有用。"),
    bullet("GVZ对5日很强；SLV IV在2026及中期预测中显著增强。"),
    bullet("GEPU和GPR在当前样本平均条件贡献为负，不进入最终模型。"),
    bullet("raw IV直接预测普遍很差；校准或作为特征后才有价值。"),
    bullet("40日raw GARCH可以作研究基准，但不能统一替代HAR。"),
    heading("7.5 最终双模型架构", 2),
    table(
      ["角色", "模型", "目的"],
      [
        ["主模型", "HAR-MSE-log + Macro + GVZ + SLV IV + US EPU", "利用全样本显示较强的跨市场与期权前瞻信息"],
        ["稳健模型", "HAR-QLIKE + Macro + US EPU", "不使用IV；基于截至2025统一结构选择，防止自证和状态漂移"],
        ["控制组", "Persistence / raw IV / GARCH", "判断收益是否只是波动持续、市场报价或GARCH均值回归"],
      ],
      [1600, 4400, 4100]
    ),
    ...imageBlock("model_qlike_comparison.png", 650, 375, "图2  全样本QLIKE相对Persistence。主/稳健模型在三个期限均明显低于100%。"),
    heading("8. 模型效果评价", 1),
    heading("8.1 最终主/稳健模型的全样本表现", 2),
    table(
      ["期限", "模型", "QLIKE", "RMSE", "OOS R²", "方向"],
      [
        ["5日", "主模型", "0.1681", "0.4953", "0.4785", "74.23%"],
        ["5日", "稳健模型", "0.1930", "0.5318", "0.3990", "74.23%"],
        ["20日", "主模型", "0.2890", "0.6072", "0.4876", "76.17%"],
        ["20日", "稳健模型", "0.2990", "0.6208", "0.4645", "75.53%"],
        ["40日", "主模型", "0.3921", "0.7059", "0.4823", "76.47%"],
        ["40日", "稳健模型", "0.3842", "0.6963", "0.4962", "78.28%"],
      ],
      [800, 2500, 1300, 1300, 1500, 1400]
    ),
    heading("8.1A 最终预测模型包的扩展五年样本评价", 2),
    paragraph(
      "为使模型交付与五年策略研究使用完全相同的输入范围，独立模型包采用2018年起的扩展特征表，并用2021年5月至2026年8月的逐月Walk-forward预测评价。下表与上一表样本起点不同，因此数值不可直接当作同一实验重复；它是交付模型在最终策略数据口径下的复现指标。"
    ),
    table(
      ["期限", "模型", "样本", "QLIKE", "RMSE", "OOS R²", "方向"],
      [
        ["5日", "主模型", "861", "0.1381", "0.4587", "51.01%", "73.87%"],
        ["5日", "稳健模型", "861", "0.1532", "0.5018", "41.38%", "70.96%"],
        ["20日", "主模型", "829", "0.2006", "0.5332", "53.12%", "76.36%"],
        ["20日", "稳健模型", "829", "0.2052", "0.5583", "48.59%", "72.50%"],
        ["40日", "主模型", "786", "0.2451", "0.5901", "55.40%", "76.34%"],
        ["40日", "稳健模型", "786", "0.2501", "0.6068", "52.84%", "74.17%"],
      ],
      [750, 2100, 900, 1150, 1150, 1400, 1400]
    ),
    paragraph(
      "截至2026-08-24重新拟合后，冻结样本的主模型年化波动率预测为14.22%/13.60%/13.29%，稳健模型为13.94%/14.19%/14.19%（对应5/20/40日）；六个方向均高于当日11.82%的当前RV。页面将这两条期限结构与经验10%—90%误差区间一起展示。"
    ),
    heading("8.2 指标如何共同理解", 2),
    table(
      ["指标", "专业含义", "本项目用法"],
      [
        ["QLIKE", "对条件方差预测的一致损失", "主选模指标；尤其防止高波动低估"],
        ["log-RV RMSE/MAE", "对数方差点预测误差", "检查典型误差大小和稳健性"],
        ["OOS R²", "相对Persistence的平方误差改善", "判断模型是否真的超过简单持续性"],
        ["方向准确率", "预测未来RV相对当前RV上升/下降是否同向", "只作辅助；不等于期权交易胜率"],
        ["DM-HLN/FDR", "比较损失差异并控制重叠目标、多重检验", "防止把噪声冠军当真"],
      ],
      [1600, 3800, 4700]
    ),
    heading("8.3 不能夸大的地方", 2),
    bullet("2026时间验证段已被项目反复查看，不再是完全 untouched holdout。"),
    bullet("40日不同模型与时期的排名波动较大，长期状态退化仍存在。"),
    bullet("控制实验组合很多；全样本冠军利用了2026已知信息，只能称描述性候选。"),
    bullet("预测指标较好不等于交易利润较好，v4直接盈亏模型就是反例。"),
    heading("9. 从预测到期权利润", 1, { pageBreakBefore: true }),
    heading("9.1 文献给出的核心共识", 2),
    bullet("实现方差预测的统计优势可以通过期权定价和Delta对冲转化为利润，但必须以成本后增量P&L评价。"),
    bullet("IV不是未来RV的无偏预测，它包含波动率风险溢价、尾部保险与流动性。"),
    bullet("长期无条件买跨式通常负收益；选择性卖波动率可能收取VRP，但承担Peso与跳跃风险。"),
    bullet("黄金的上涨/下跌波动和VRP并不完全对称，未来应拆分半方差和跳跃。"),
    bullet("历史高胜率不能替代尾部压力测试；真实Bid/Ask和合约特定Greeks不可省略。"),
    heading("9.2 策略信号", 2),
    paragraph(
      "同期限主模型和稳健模型的预测方差分别加入只使用已到期历史估计的VRP，得到公允波动率；当两者均比市场ATM IV低至少1.5 vol points、流动性有效且无显著Jump时产生短波动率信号。随后在下一交易日的真实同步盘口卖出ATM Call和Put，并建立期货Delta对冲。"
    ),
    heading("9.3 为什么不直接用IV预测并同时交易IV", 2),
    paragraph(
      "如果模型只复制当前AU IV，再用该模型说AU IV贵或便宜，就容易自证。项目采用三道隔离：预测目标始终是未来实现方差；含IV信息只放在主模型；无IV稳健模型必须同意。此外，交易IV和公允波动率之间还隔着因果VRP估计。"
    ),
    heading("10. 策略研究的完整演进", 1),
    table(
      ["版本/阶段", "主要改变", "核心结果", "保留的教训"],
      [
        ["v1 理想日线", "模型择时短跨、每日Delta、日线开盘/收盘与假设滑点", "21笔，净利润770,100，Sharpe 1.17", "模型择时优于Persistence和无脑卖；但执行过于理想"],
        ["v2 六项保守改进", "90%上界、尾部分解、直接盈亏下界、铁蝶、真实五腿盘口、逐日保证金", "只剩2笔，净利润3,418，Sharpe 0.076", "保护翼真实价差可能吞掉大部分优势；过度过滤会失去样本"],
        ["v3 真实期权盘口", "回到旧波动率信号，ATM跨式真实bid/ask，1×/3×深度", "1×净利润242,910，Sharpe 0.791", "真实一档容量显著降低手数；但旧信号仍有经济价值"],
        ["v4 直接预测利润", "Ridge/RF/HGB直接预测每手最终净P&L", "预测OOS R² 0.53—0.59，但策略亏损", "预测标签拟合好不等于选出来的交易赚钱；40日少数错误很致命"],
        ["v5 利润增强", "1×—12×容量、对冲频率、Edge、价差、动态仓位等38方案", "5×为容量甜点；窄盘口≤25% Sharpe高", "最高利润方案受单笔集中和理想中间对冲影响，不宜直接上线"],
        ["v6 五年真实期货top-book", "扩展日历窗口；所有期货调仓改真实bid1/ask1与可见量", "窄盘口净利润402,270，Sharpe1.015", "真实执行使利润下降但可信度上升；2021—2023仍无交易"],
        ["v7 窄盘口微调", "11个预先冻结变体", "10%风险预算净利润464,830，Sharpe1.061；20/40 Sharpe1.143但仅7笔", "简单提高仓位更可实施；不采用小样本100%胜率冠军"],
        ["v8 风险×对冲网格", "6档风险预算×1/2/3日对冲共18组", "10%/2日最高Sharpe1.085，净利润509,760", "两日频率有利；12%以后被容量封顶；最终选择10%"],
      ],
      [1500, 3100, 2600, 2900],
      { fontSize: 13 }
    ),
    heading("10.1 v2为什么没有替代旧版", 2),
    paragraph(
      "v2是风险结构上更安全的铁蝶，但远虚值保护翼在真实市场里可能极宽。两笔交易的显式手续费只有442元，而从中间价跨到真实可成交价的隐含成本达25,080元，是显式成本的56.7倍。它证明“有保护翼”不等于“保护翼可经济成交”。"
    ),
    heading("10.2 v4为什么统计好看却亏钱", 2),
    paragraph(
      "直接P&L模型对每手利润大小有较高解释度，但最终组合选择受期限尺度、少数40日大额标签和方向错误支配。研究因此放弃让小样本利润模型独立搜索全部交易，保留它作为未来二级排序或仓位工具的可能。"
    ),
    heading("10.3 真实买卖价的重要性", 2),
    paragraph(
      "V6用真实期货top-book替换日线收盘＋1 tick后，V3 5×、窄盘口和最高利润方案的利润分别下降14.1%、27.3%、47.7%。这不是模型变差，而是过去被忽略的真实买卖价差、按可执行侧估值和一档数量限制被计入。"
    ),
    heading("11. 最终策略与结果", 1),
    callout(
      "冻结最终策略",
      "含IV主模型＋无IV稳健模型一致做空；VRP因果校准；Jump否决；ATM跨式；相对价差≤25%；期权5×显示深度容量假设；单笔风险预算10%；组合风险上限30%；每2个交易日按真实期货bid1/ask1进行Delta对冲。",
      C.gold,
      C.paleGold
    ),
    metricCards([
      { value: "¥509,760", label: "净利润", color: C.green, fill: C.paleGreen },
      { value: "1.084936", label: "Sharpe", color: C.blue, fill: C.paleBlue },
      { value: "-0.909%", label: "最大回撤", color: C.red, fill: "F7EAEA" },
      { value: "10/12", label: "盈利交易", color: C.gold, fill: C.paleGold },
    ]),
    ...imageBlock("final_equity_drawdown.png", 650, 379, "图3  最终策略累计权益与回撤。"),
    ...imageBlock("grid_sharpe_heatmap.png", 650, 425, "图4  最终参数所在的18组风险预算×对冲频率网格。"),
    heading("11.1 与原始窄盘口策略比较", 2),
    table(
      ["指标", "原始7.5%＋每日", "最终10%＋每2日", "变化"],
      [
        ["净利润", "402,270", "509,760", "+107,490 / +26.7%"],
        ["Sharpe", "1.0149", "1.0849", "+0.0701"],
        ["最大回撤", "-1.0697%", "-0.9092%", "改善0.1605个百分点"],
        ["胜率", "75.0%", "83.3%", "+8.3个百分点"],
        ["期货对冲周转", "100手", "72手", "-28%"],
        ["期货订单", "86", "56", "-34.9%"],
      ],
      [2300, 2500, 2500, 2800]
    ),
    heading("11.2 为什么不是12%＋每2日", 2),
    paragraph(
      "12%组合利润为513,300元，仅高3,540元；Sharpe略低于10%组合。12%、15%、18%在同一频率下完全相同，是期权5×容量和真实期货一档深度封顶。用户最终选择10%，意味着在历史上几乎不牺牲利润，同时保持较低名义仓位扩张能力。"
    ),
    heading("12. 局限性", 1),
    heading("12.1 样本与模型选择", 2),
    bullet("日历窗口超过五年，但真正交易只有2024—2026的12笔，2021—2023没有信号。"),
    bullet("多个模型、变量块和策略参数在同一历史上比较，存在数据窥探与多重检验。"),
    bullet("2026已被反复查看，不能再宣称为完全未触碰测试集。"),
    bullet("Bootstrap区间和高正收益概率建立在历史交易簇可重采样的假设上，不能覆盖未知制度变化。"),
    heading("12.2 执行与风险", 2),
    bullet("期权5×深度是容量假设，不是L2-L5逐档真实成交回放。"),
    bullet("期权日度标记不是每天真实可平仓top-book，会影响Sharpe和回撤路径。"),
    bullet("最终策略为未加翼短跨式，理论尾部风险不封顶；历史最大回撤远低于潜在压力损失。"),
    bullet("研究手续费不是用户真实经纪商账单；客户保证金和临时加保尚未精确复原。"),
    bullet("不存在排队位置、双腿不同步、撤单、拒单、断线、涨跌停和强平的完整实盘仿真。"),
    heading("12.3 数据", 2),
    bullet("GPR及部分宏观序列存在历史修订或保守发布时间近似。"),
    bullet("SLV IV来自有限成本重建，不能等同于完整无套利IV曲面。"),
    bullet("历史AU期权流动性较短，真正可交易信号只集中在近两年。"),
    heading("13. 展望与改进路线", 1),
    table(
      ["优先级", "任务", "目的", "验收条件"],
      [
        ["P0", "冻结前瞻纸面交易6—12个月", "获得真正未触碰样本", "至少20—30笔成熟交易；不改参数"],
        ["P0", "期权L2-L5与组合成交", "替换5×容量假设", "逐档价格、部分成交和未成交可重放"],
        ["P0", "尾部封顶", "把短跨式极端风险写进结构", "保护翼真实价差可接受；压力损失≤账户阈值"],
        ["P0", "真实客户保证金", "避免研究资金占用偏差", "接入账户级费率、提保和组合优惠"],
        ["P1", "嵌套Walk-forward", "降低参数与模型选择偏差", "外层只评价，内层才选参数；到期簇purge/embargo"],
        ["P1", "直接预测P&L分布", "改善交易排序和仓位", "按期限标准化；预测损失概率和下分位数"],
        ["P1", "半方差/Jump/Skew", "识别黄金上下行尾部不对称", "在独立OOS提高收益或降低尾部"],
        ["P2", "事件新闻重放", "处理状态切换与卖方否决", "固定提示词；15/30/60分钟发现延迟回测"],
        ["P2", "实盘执行网关", "从研究程序走向可控生产", "限价、撤单、断线、审计、风控和人工确认"],
      ],
      [900, 2600, 3100, 3500],
      { fontSize: 14 }
    ),
    heading("14. 从零开始到当前的完整工作流", 1, { pageBreakBefore: true }),
    numbered("读取初始PPT与指标讨论材料，明确预测对象是未来RV而不是金价方向。"),
    numbered("把约50个字段拆成数据字典、目标、预测特征、执行字段、过滤器与输出，避免大回归。"),
    numbered("建立本地Python项目、配置、环境变量、Parquet数据层和单元测试。"),
    numbered("小区间估价和连接测试后，从TQSDK、Databento、FRED、官方日历及GPR作者站下载数据。"),
    numbered("统一UTC/上海时间、夜盘交易日、代表合约、数据可得时间和陈旧规则。"),
    numbered("用5分钟收益计算RV、BV、Jump、RQ、HAR日/周/月特征和5/20/40日前向目标。"),
    numbered("构造COMEX非重叠波动、宏观变化、事件计数、GVZ、SLV IV、US/Global EPU和GPR。"),
    numbered("先训练Ridge-HAR-X基线，再加入QLIKE与HARQ，严格Walk-forward并保存模型包。"),
    numbered("运行HAR/HARQ/GARCH×MSE/QLIKE×256子集控制实验，对照Persistence与raw IV。"),
    numbered("识别IV自证风险，确定含IV主模型＋无IV稳健模型双确认，GARCH只作对照。"),
    numbered("检索27项波动率预测、期权收益、VRP、商品期权和交易所资料，设计ATM跨式、VRP校准与Delta对冲。"),
    numbered("完成v1理想日线经济检验，确认模型择时优于Persistence和无脑卖。"),
    numbered("在v2加入预测上界、尾部分解、直接P&L、铁蝶、真实五腿盘口和保证金；发现保护翼价差和过度过滤。"),
    numbered("v3回到旧信号但使用真实期权Bid/Ask；v4测试直接P&L模型并因策略亏损淘汰。"),
    numbered("v5测试1×—12×容量和38个利润增强方案，识别5×容量甜点和窄盘口优势。"),
    numbered("v6扩展到五年日历窗口，并把所有期货开仓、调仓、到期平仓替换为真实top-book。"),
    numbered("v7围绕窄盘口做11项冻结微调，确认10%风险预算是简单有效改进。"),
    numbered("v8做6×3风险预算/对冲频率网格，最终选择10%＋每2日；导出冻结程序、订单、审计和文档。"),
    numbered("把预测层另行冻结为六个模型文件，提供CSV/Parquet调用器、输入模板、本地可视化页面、模型哈希和参考预测零误差校验。"),
    heading("15. 工程质量与可复现性", 1),
    bullet("最终全项目pytest：81项通过，只有joblib无法读取物理核心数的无害警告。"),
    bullet("最终单策略产生104行订单/结算记录、56笔期货订单，invalid=0。"),
    bullet("每次重要策略升级使用独立源码/配置/输出目录，不覆盖旧版本。"),
    bullet("冻结清单保存配置、源码、数据和结果SHA-256；最终包另有package_manifest.json。"),
    bullet("最终配置具有防篡改校验；若修改10%、2日、25%或关键执行参数，程序拒绝沿用同一版本。"),
    heading("16. 研究参考文献与规则来源", 1, { pageBreakBefore: true }),
    paragraph("以下为项目文献检索中直接影响方法与策略设计的核心来源。链接用于后续交接核验。", { after: 120 }),
    ...[
      "1. Engle, Hong & Kane (1990). Valuation of Variance Forecast with Simulated Option Markets. NBER w3350. https://www.nber.org/papers/w3350",
      "2. Engle, Kane & Noh (1993). Index-Option Pricing with Stochastic Volatility and the Value of Accurate Variance Forecasts. NBER w4519. https://www.nber.org/papers/w4519",
      "3. Noh, Engle & Kane (1993). A Test of Efficiency for the S&P Index Option Market Using Variance Forecasts. NBER w4520. https://www.nber.org/papers/w4520",
      "4. Bandi, Russell & Yang (2008). Realized Volatility Forecasting and Option Pricing. Journal of Econometrics. https://doi.org/10.1016/j.jeconom.2008.09.002",
      "5. Corsi (2009). A Simple Approximate Long-Memory Model of Realized Volatility. Journal of Financial Econometrics.",
      "6. Bollerslev, Patton & Quaedvlieg (2016). Exploiting the Errors: A Simple Approach for Improved Volatility Forecasting. Journal of Econometrics.",
      "7. Christoffersen & Jacobs (2004). The Importance of the Loss Function in Option Valuation. Journal of Financial Economics. https://doi.org/10.1016/j.jfineco.2003.02.001",
      "8. Patton (2011). Volatility Forecast Comparison Using Imperfect Volatility Proxies. Journal of Econometrics.",
      "9. Coval & Shumway (2001). Expected Option Returns. Journal of Finance. https://doi.org/10.1111/0022-1082.00352",
      "10. Bakshi & Kapadia (2003). Delta-Hedged Gains and the Negative Market Volatility Risk Premium. Review of Financial Studies.",
      "11. Goyal & Saretto (2009). Cross-section of Option Returns and Volatility. Journal of Financial Economics.",
      "12. Carr & Wu (2009). Variance Risk Premiums. Review of Financial Studies. https://doi.org/10.1093/rfs/hhn038",
      "13. Broadie, Chernov & Johannes (2009). Understanding Index Option Returns. Review of Financial Studies. https://doi.org/10.1093/rfs/hhp032",
      "14. Bondarenko (2014). Why Are Put Options So Expensive? Quarterly Journal of Finance.",
      "15. Duarte & Jones (2007). The Price of Market Volatility Risk. Journal of Econometrics.",
      "16. Bollerslev, Tauchen & Zhou (2009). Expected Stock Returns and Variance Risk Premia. Review of Financial Studies.",
      "17. Bollerslev, Gibson & Zhou (2011). Dynamic Estimation of Volatility Risk Premia. Journal of Econometrics.",
      "18. Aït-Sahalia, Karaman & Mancini (2020). The Term Structure of Variance Swaps and Risk Premia. Journal of Econometrics.",
      "19. Dew-Becker et al. (2017). The Price of Variance Risk. Journal of Financial Economics.",
      "20. Andersen, Fusari & Todorov (2015). The Risk Premia Embedded in Index Options.",
      "21. Carr & Wu (2016). Analyzing Volatility Risk and Risk Premium in Option Contracts. Journal of Financial Economics.",
      "22. Israelov & Kelly (2017). Forecasting the Distribution of Option Returns.",
      "23. Israelov & Tummala (2018). Being Right Is Not Enough: Buying Options to Profit from Higher Volatility.",
      "24. CME Group (2012). Volatility Trading for Gold: Hedging Global Instability.",
      "25. Tee & Ting (2017). Variance Risk Premiums of Commodity ETFs. Journal of Futures Markets. https://doi.org/10.1002/fut.21802",
      "26. Indriawan et al. (2019). Bad Volatility Is Not Always Bad: Evidence from Precious Metals.",
      "27. 上海期货交易所：黄金期权合约、风险管理资料、黄金期货业务细则及历次保证金通知。https://www.shfe.com.cn/",
    ].map((text) => paragraph(text, { after: 70, line: 290, run: { size: 17 } })),
    pageBreak(),
    heading("附录A：完整假设判定表", 1),
    table(
      ["问题", "最终判定", "后续动作"],
      [
        ["是否把全部50个字段放进模型", "否；字段分层，核心结构小型化", "只通过分块OOS实验引入新变量"],
        ["是否统一用HARQ", "否；短中期无稳定增益", "40日保留挑战者记录"],
        ["是否让GARCH替代HAR", "否；40日有用但无法统一", "作为独立基准和未来集成候选"],
        ["是否加入IV", "主模型加入，稳健模型排除", "继续检查主/稳健分歧"],
        ["是否直接预测利润", "本轮失败，不独立下单", "样本扩大后按期限预测分布"],
        ["是否使用保护翼", "v2静态风险更安全，但真实价差过高", "未来按风险+流动性动态选翼"],
        ["是否选择20/40-only", "暂不；只有7笔且100%胜率可疑", "作为前瞻观察组"],
        ["是否选择12%风险预算", "否；只多3,540元且名义预算更高", "最终10%"],
      ],
      [4000, 3100, 3000],
      { fontSize: 15 }
    ),
    heading("附录B：失败与回退记录", 1),
    bullet("HARQ没有在5/20日改善，因此不把“更复杂”包装成“更先进”。"),
    bullet("GEPU/GPR平均条件增益为负，因此没有因为数据来之不易就强行保留。"),
    bullet("v2安全结构只剩2笔、保护翼价差巨大，因此保留但不替代收益主线。"),
    bullet("v4直接P&L模型预测统计不错但策略亏损，因此淘汰独立交易用途。"),
    bullet("8×/12×深度没有超过5×，说明容量与风险限制比名义深度更重要。"),
    bullet("利润型组合在真实期货top-book后利润显著下降，说明理想执行不能作为上线依据。"),
    bullet("V7保守组合叠加多重过滤后表现最差，说明规则堆叠会损害有效机会。"),
    bullet("V8三日对冲在所有风险预算下明显差，说明过度降低对冲频率会暴露方向风险。"),
    heading("附录C：核心文件地图", 1),
    table(
      ["文件/目录", "内容"],
      [
        ["config.yaml", "数据、特征、模型、控制实验和基础策略主配置"],
        ["src/au_rv/features/", "RV、Jump、宏观、外部指标与目标构造"],
        ["src/au_rv/models/", "HAR/HARQ/GARCH、Walk-forward、QLIKE、控制实验"],
        ["src/au_rv/final_model/", "最终主/稳健六模型的训练、冻结校验与推理接口"],
        ["src/au_rv/strategy/", "Black-76、期权机会、VRP与基线策略"],
        ["src/au_rv/strategy_v6/", "五年与真实期货top-book执行引擎"],
        ["data/outputs/control_experiment_*", "模型控制变量预测、指标、排名和冠军"],
        ["data/outputs/strategy_v2/", "六项保守改进与铁蝶真实盘口结果"],
        ["data/outputs/strategy_v3_real_quotes/", "真实期权盘口跨式结果"],
        ["data/outputs/strategy_v4_direct_pnl/", "直接盈亏模型结果"],
        ["data/outputs/strategy_v5_profit_enhancement/", "容量与利润增强实验"],
        ["data/outputs/strategy_v6_real_topbook/", "五年真实期货top-book三策略结果"],
        ["data/outputs/strategy_v7_narrow_tuning/", "窄盘口11种微调"],
        ["data/outputs/strategy_v8_narrow_risk_hedge_grid/", "18组风险/对冲网格"],
        ["data/outputs/final_strategy_10pct_2d/", "最终单策略冻结结果"],
        ["models/final_prediction_model/", "5/20/40日主模型与无IV稳健模型六个冻结文件"],
        ["ui/final_model_dashboard/", "仅本机运行的预测期限结构与历史评价页面"],
      ],
      [4700, 5400],
      { fontSize: 15 }
    ),
    heading("附录D：最终结论", 1),
    paragraph(
      "项目最重要的成果不是找到一个历史收益最高的参数，而是建立了一条从因果数据、波动率预测、对照实验、经济检验到真实盘口执行的可审计链条。交付物把预测模型与交易策略拆成两个独立可验证程序包：前者负责5/20/40日RV预测与双模型诊断，后者负责VRP、期权IV、盘口、仓位和对冲。最终10%风险预算＋每2日对冲是当前历史证据下最具资本效率的候选，但它仍应被当作前瞻纸面交易的冻结起点，而不是已经证明的收益机器。"
    )
  );
  return new Document(s);
}

async function main() {
  const manualPath = path.join(OUT, "沪金期权最终策略说明书_10pct风险预算_每2日对冲_20260831.docx");
  const reportPath = path.join(OUT, "沪金期权波动率预测项目完整研究报告_20260831.docx");
  fs.writeFileSync(manualPath, await Packer.toBuffer(buildStrategyManual()));
  fs.writeFileSync(reportPath, await Packer.toBuffer(buildResearchReport()));
  console.log(manualPath);
  console.log(reportPath);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
