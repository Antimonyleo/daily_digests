import test from 'node:test';
import assert from 'node:assert/strict';
import {backupReminderDue} from './app.mjs';
test('backup reminders allow the first week, flag stale exports, and never invent a backup',()=>{
  const now=Date.parse('2026-10-16T12:00:00Z'),day=86400000;
  assert.equal(backupReminderDue({firstOpen:String(now-day),now}),false);
  assert.equal(backupReminderDue({firstOpen:String(now-7*day),now}),true);
  assert.equal(backupReminderDue({lastExport:'2026-10-01T12:00:00Z',firstOpen:String(now),now}),true);
  assert.equal(backupReminderDue({lastExport:'invalid',firstOpen:String(now),now}),false);
  assert.equal(backupReminderDue({now}),false);
});
