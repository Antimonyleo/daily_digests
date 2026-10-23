import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const template=readFileSync(new URL('../templates/setup.html.j2',import.meta.url),'utf8');
const block=template.match(/  const timezoneInput =[^]*?catch \(_\) \{\}/)?.[0];
assert.ok(block, 'actual Setup timezone behavior');
function timezone(firstRun, configured, detected) {
 const input={value:configured,dataset:{detectTimezone:String(firstRun)}};
 vm.runInNewContext(block,{document:{getElementById:()=>input},Intl:{DateTimeFormat:()=>({resolvedOptions:()=>({timeZone:detected})})}});
 return input.value;
}
test('saving an existing profile in another browser timezone preserves its digest date',()=>{
 assert.equal(timezone(false,'UTC','America/Phoenix'),'UTC');
 assert.equal(timezone(false,'Asia/Shanghai','America/Phoenix'),'Asia/Shanghai');
});
test('first-time setup adopts a detected browser timezone and keeps fallback if absent',()=>{
 assert.equal(timezone(true,'UTC','America/Phoenix'),'America/Phoenix');
 assert.equal(timezone(true,'UTC',undefined),'UTC');
});
