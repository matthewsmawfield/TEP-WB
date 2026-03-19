const fs = require('fs');
const path = require('path');
const { createManuscriptContext, injectPlaceholders } = require('../site/manuscript-data.js');

const COMPONENTS_DIR = path.join(__dirname, '../site/components');
const OUTPUT_FILE = path.join(__dirname, '../15manuscript-tep-wb.md');

const components = [
    '1_abstract.html',
    '1_introduction.html',
    '2_theory.html',
    '3_methodology.html',
    '4_results.html',
    '5_environment.html',
    '6_discussion.html',
    '7_conclusion.html',
    '8_references.html'
];

function convertTable(tableHtml) {
    let result = '';
    // Extract caption
    const capMatch = tableHtml.match(/<caption>([\s\S]*?)<\/caption>/);
    if (capMatch) {
        let cap = capMatch[1].trim();
        cap = cap.replace(/<strong>(.*?)<\/strong>/g, '**$1**');
        cap = cap.replace(/<em>(.*?)<\/em>/g, '*$1*');
        result += cap + '\n\n';
    }
    // Extract rows
    const rows = [];
    const rowRe = /<tr>([\s\S]*?)<\/tr>/g;
    let rm;
    while ((rm = rowRe.exec(tableHtml)) !== null) {
        const cells = [];
        const cellRe = /<t[hd]>([\s\S]*?)<\/t[hd]>/g;
        let cm;
        while ((cm = cellRe.exec(rm[1])) !== null) {
            cells.push(cm[1].trim());
        }
        if (cells.length > 0) rows.push(cells);
    }
    if (rows.length > 0) {
        result += '| ' + rows[0].join(' | ') + ' |\n';
        result += '| ' + rows[0].map(() => '---').join(' | ') + ' |\n';
        for (let i = 1; i < rows.length; i++) {
            result += '| ' + rows[i].join(' | ') + ' |\n';
        }
    }
    return result;
}

function convertFigure(figHtml) {
    let result = '';
    const imgMatch = figHtml.match(/<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*/);
    const capMatch = figHtml.match(/<figcaption>([\s\S]*?)<\/figcaption>/);
    if (imgMatch) {
        const figurePath = imgMatch[1].startsWith('figures/')
            ? `results/${imgMatch[1]}`
            : imgMatch[1];
        result += `![${imgMatch[2]}](${figurePath})\n`;
    }
    if (capMatch) {
        let cap = capMatch[1].trim();
        cap = cap.replace(/<strong>(.*?)<\/strong>/g, '**$1**');
        cap = cap.replace(/<em>(.*?)<\/em>/g, '*$1*');
        result += cap + '\n';
    }
    return result;
}

function htmlToMarkdown(html) {
    let md = html;

    // Convert tables first (before stripping other tags)
    md = md.replace(/<table[^>]*>([\s\S]*?)<\/table>/g, (m, inner) => convertTable(inner));

    // Convert figures
    md = md.replace(/<figure>([\s\S]*?)<\/figure>/g, (m, inner) => convertFigure(inner));

    // Preserve page breaks
    md = md.replace(/<div style="page-break-after:\s*always;?"><\/div>/g, '@@PAGE_BREAK@@');

    // Strip div wrappers (class="equation", class="section", etc.)
    md = md.replace(/<div[^>]*>/g, '');
    md = md.replace(/<\/div>/g, '');

    // Restore page breaks
    md = md.replace(/@@PAGE_BREAK@@/g, '<div style="page-break-after: always;"></div>');

    // Convert headers
    md = md.replace(/<h2>(.*?)<\/h2>/g, '## $1');
    md = md.replace(/<h3>(.*?)<\/h3>/g, '### $1');

    // Convert inline markup
    md = md.replace(/<strong>(.*?)<\/strong>/g, '**$1**');
    md = md.replace(/<em>(.*?)<\/em>/g, '*$1*');

    // Convert paragraphs (handle multi-line and class attributes)
    md = md.replace(/<p[^>]*>([\s\S]*?)<\/p>/g, (m, content) => content.trim() + '\n');

    // Convert lists
    md = md.replace(/<[ou]l>/g, '');
    md = md.replace(/<\/[ou]l>/g, '');
    md = md.replace(/<li>([\s\S]*?)<\/li>/g, (m, content) => '- ' + content.trim());

    // Convert HTML entities
    md = md.replace(/&mdash;/g, '\u2014');
    md = md.replace(/&ndash;/g, '\u2013');
    md = md.replace(/&amp;/g, '&');
    md = md.replace(/&lt;/g, '<');
    md = md.replace(/&gt;/g, '>');
    md = md.replace(/&nbsp;/g, ' ');

    // Clean up whitespace: collapse 3+ blank lines to 2
    md = md.replace(/\n{3,}/g, '\n\n');
    // Trim leading/trailing whitespace from each line
    md = md.split('\n').map(l => l.trim()).join('\n');

    return md.trim();
}

let markdown = '# The Temporal Equivalence Principle: Density-Dependent Screening in Gaia DR3 Wide Binaries\n\n';
const injectionContext = createManuscriptContext();

components.forEach(file => {
    const filePath = path.join(COMPONENTS_DIR, file);
    if (fs.existsSync(filePath)) {
        const html = injectPlaceholders(fs.readFileSync(filePath, 'utf8'), injectionContext, { failOnUnresolved: true });
        markdown += htmlToMarkdown(html) + '\n\n';
    } else {
        console.warn(`Warning: Component ${file} not found.`);
    }
});

fs.writeFileSync(OUTPUT_FILE, markdown);
console.log(`Generated ${OUTPUT_FILE}`);
