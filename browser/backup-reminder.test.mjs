import test from 'node:test';
import assert from 'node:assert/strict';
import {backupReminderDue, quietStorageProtection} from './app.mjs';
test('backup reminders allow the first week, flag stale exports, and never invent a backup',()=>{
  const now=Date.parse('2026-10-16T12:00:00Z'),day=86400000;
  assert.equal(backupReminderDue({firstOpen:String(now-day),now}),false);
  assert.equal(backupReminderDue({firstOpen:String(now-7*day),now}),true);
  assert.equal(backupReminderDue({lastExport:'2026-10-01T12:00:00Z',firstOpen:String(now),now}),true);
  assert.equal(backupReminderDue({lastExport:'invalid',firstOpen:String(now),now}),false);
  assert.equal(backupReminderDue({now}),false);
});
test('storage that a browser may clear after a week gets a reminder after three days',()=>{
  const now=Date.parse('2026-10-16T12:00:00Z'),day=86400000;
  assert.equal(backupReminderDue({firstOpen:String(now-3*day),now}),false);
  assert.equal(backupReminderDue({firstOpen:String(now-3*day),now,days:3}),true);
});
test('storage protection is requested quietly except where Firefox would prompt',()=>{
  assert.equal(quietStorageProtection('Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15'),true);
  assert.equal(quietStorageProtection('Mozilla/5.0 (X11; Linux x86_64) Chrome/141.0 Safari/537.36'),true);
  assert.equal(quietStorageProtection('Mozilla/5.0 (X11; Linux x86_64; rv:144.0) Gecko/20100101 Firefox/144.0'),false);
});
