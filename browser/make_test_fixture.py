"""Create a fictional, isolated browser acceptance backup and RSS feed."""
import argparse
import os,sys,json,base64,html
from pathlib import Path
from datetime import datetime,timezone,timedelta
from email.utils import format_datetime
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'fixture-state').exists():
 parser.error('Choose a fresh output directory; fixture-state already exists')
STATE=OUT/'fixture-state'
STATE.mkdir(exist_ok=True)
os.chdir(STATE)
sys.path.insert(0,str(ROOT/'src'))
for k,v in {'DB_PATH':'data/digest.db','PROFILE_PATH':'data/profile.yaml','SOURCES_PATH':'data/acceptance-sources.yaml','LLM_BACKEND':'extractive','LLM_API_KEY':'','RESEND_API_KEY':'','DIGEST_TO':'','USER_TZ':'UTC','INCLUDE_OPPORTUNITIES':'false','INCLUDE_EVENTS':'true','INCLUDE_INDUSTRY':'false','INCLUDE_AI':'false','INCLUDE_REGULATORY':'false','INCLUDE_WORLD':'false','TOP_RESEARCH':'10'}.items(): os.environ[k]=v
DATA=STATE/'data';DATA.mkdir(exist_ok=True)
import yaml
profile={'name':'Browser Acceptance','bio':'Engineering programmable DNA assemblies, synthetic condensates and transferable molecular design and prediction methods.','keywords':['DNA nanotechnology','DNA origami','protein design','de novo protein design','synthetic condensates','molecular machine learning'],'context_keywords':['native protein signaling'],'interest_weights':{'DNA nanotechnology':1.5,'protein design':1.2},'downweight':['clinical case report']}
(DATA/'profile.yaml').write_text(yaml.safe_dump(profile))
(DATA/'opportunities.yaml').write_text(yaml.safe_dump({'description':'Fictional molecular design researcher','career_stage':'postdoctoral researcher','institution_type':'university','country':'United States','applicant_role':'individual','event_formats':['online'],'minimum_lead_days':7}))
sources={'research':[{'name':'Nature','kind':'rss','url':'https://acceptance.invalid/research.rss','quality_tier':'top','prestige_score':1.0,'impact_floor':7.0}]}
(DATA/'acceptance-sources.yaml').write_text(yaml.safe_dump(sources))
(STATE/'.env').write_text(''.join(f'{k}={v}\n' for k,v in os.environ.items() if k in {'DB_PATH','PROFILE_PATH','SOURCES_PATH','LLM_BACKEND','USER_TZ','INCLUDE_OPPORTUNITIES','INCLUDE_EVENTS','INCLUDE_INDUSTRY','INCLUDE_AI','INCLUDE_REGULATORY','INCLUDE_WORLD','TOP_RESEARCH'}))
from dailydigest.models import Item
from dailydigest import store
now=datetime.now(timezone.utc)
items=[Item(source='Nature',section='research',external_id='seed-'+str(i),url='https://example.invalid/seed/'+str(i),title=title,abstract='We introduce a transferable engineering method for designing programmable biomolecular assemblies.',authors='Fixture Author',published_at=now-timedelta(days=2)) for i,title in enumerate(['Programmable DNA origami assembly design','De novo protein binder engineering','Machine learning for synthetic condensate design'],1)]
items.append(Item(source='EMBL scientific events',section='events',external_id='calendar-fixture',url='https://example.invalid/event',title='Molecular design methods workshop',abstract='A fictional acceptance event.',published_at=now,metadata={'record_type':'event','status':'open','format':'online','official_source':True,'event_start':(now+timedelta(days=30)).date().isoformat(),'event_end':(now+timedelta(days=31)).date().isoformat()}))
store.upsert_items(items)
with store.session_scope() as s:
 rows=s.query(store.ItemRow).order_by(store.ItemRow.id).all()
 ids=[r.id for r in rows]
 for r in rows:r.summary='Key point: Transferable engineering method. Why read: Molecular design. Caveat: Fictional test input.'
store.write_digest((now-timedelta(days=2)).date().isoformat(),[('R1',ids[0],.8),('R2',ids[1],.82),('E1',ids[3],.9)])
store.write_digest(now.date().isoformat(),[('R1',ids[2],.85),('E1',ids[3],.9)])
store.set_bookmark(ids[0],True)
store._ENGINE.dispose()
files={str(p.relative_to(STATE)):base64.b64encode(p.read_bytes()).decode() for p in DATA.rglob('*') if p.is_file()}
files['.env']=base64.b64encode((STATE/'.env').read_bytes()).decode()
(OUT/'fixture-backup.json').write_text(json.dumps({'format':'dailydigest-browser-v1','files':files}))
corpus=json.loads((ROOT/'tests/fixtures/research_scope_benchmark.json').read_text())
rows=corpus['cases'] if isinstance(corpus,dict) and 'cases' in corpus else corpus
if isinstance(rows,dict): rows=next(v for v in rows.values() if isinstance(v,list))
xml=['<?xml version="1.0"?><rss version="2.0"><channel><title>Nature fictional acceptance feed</title><link>https://acceptance.invalid/</link><description>Fictional tests</description>']
for i,row in enumerate(rows):
 title=row['title'];abstract=row.get('abstract','')
 xml.append('<item><guid>browser-case-'+str(i)+'</guid><link>https://acceptance.invalid/paper/'+str(i)+'</link><title>'+html.escape(title)+'</title><description>'+html.escape(abstract)+'</description><pubDate>'+format_datetime(now)+'</pubDate><author>Fixture Author</author></item>')
xml.append('</channel></rss>')
(OUT/'research-feed.xml').write_text(''.join(xml))
(OUT/'fixture-ids.json').write_text(json.dumps({'research_ids':ids[:3],'calendar_id':ids[3],'digest_id':now.date().isoformat(),'papers':len(rows)}))
print(json.dumps({'papers':len(rows),'ids':ids}))
