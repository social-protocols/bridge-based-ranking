
.import userEnrollment-00000.tsv userEnrollment
.import notes-00000.tsv notes
.import noteStatusHistory-00000.tsv noteStatusHistory

.import ratings-00000.tsv ratings0
.import ratings-00001.tsv ratings1
.import ratings-00002.tsv ratings2
.import ratings-00003.tsv ratings3

create table ratings as select * from ratings0 union all select * from ratings1 union all select * from ratings2 union all select * from ratings3;
drop table ratings0;
drop table ratings1;
drop table ratings2;
drop table ratings3;


create unique index ratings_note_participant on ratings(noteId, raterParticipantId);
create unique index note_id on notes(noteId);
create unique index participant_id on userEnrollment(participantId);


select modelingGroup, modelingPopulation, count(*) from userEnrollment group by 1, 2;

select version, datetime(min(createdAtMillis)/1000, "unixepoch"), datetime(max(createdAtMillis)/1000, "unixepoch"), count(*) from ratings group by 1;


create table noteStats2 as
select 
	noteId
	, count(*) as cnt
	, min(createdAtMillis) minCreatedAtMillis
	, max(createdAtMillis) maxCreatedAtMillis
	, min(version) as minVersion
	, max(version) as maxVersion
	, sum(agree) as agree 
	, sum(disagree) as disagree 
	, sum(helpful) as helpful
	, sum(notHelpful) as notHelpful
	, sum(case when helpfulnessLevel = 'HELPFUL' then 1 else 0 end) as helpfulnessLevelHelpful 
	, sum(case when helpfulnessLevel = 'NOT_HELPFUL' then 1 else 0 end) as helpfulnessLevelNotHelpful 
	, sum(case when helpfulnessLevel = 'SOMEWHAT_HELPFUL' then 1 else 0 end) as helpfulnessLevelSomewhatHelpful 
	, sum(case when helpfulnessLevel = '' then 1 else 0 end) as helpfulnessLevelNone 
	, sum(helpfulOther) as helpfulOther
	, sum(helpfulInformative) as helpfulInformative
	, sum(helpfulClear) as helpfulClear
	, sum(helpfulEmpathetic) as helpfulEmpathetic
	, sum(helpfulGoodSources) as helpfulGoodSources
	, sum(helpfulUniqueContext) as helpfulUniqueContext
	, sum(helpfulAddressesClaim) as helpfulAddressesClaim
	, sum(helpfulImportantContext) as helpfulImportantContext
	, sum(helpfulUnbiasedLanguage) as helpfulUnbiasedLanguage
	, sum(notHelpfulOther) as notHelpfulOther
	, sum(notHelpfulIncorrect) as notHelpfulIncorrect
	, sum(notHelpfulSourcesMissingOrUnreliable) as notHelpfulSourcesMissingOrUnreliable
	, sum(notHelpfulOpinionSpeculationOrBias) as notHelpfulOpinionSpeculationOrBias
	, sum(notHelpfulMissingKeyPoints) as notHelpfulMissingKeyPoints
	, sum(notHelpfulOutdated) as notHelpfulOutdated
	, sum(notHelpfulHardToUnderstand) as notHelpfulHardToUnderstand
	, sum(notHelpfulArgumentativeOrBiased) as notHelpfulArgumentativeOrBiased
	, sum(notHelpfulOffTopic) as notHelpfulOffTopic
	, sum(notHelpfulSpamHarassmentOrAbuse) as notHelpfulSpamHarassmentOrAbuse
	, sum(notHelpfulIrrelevantSources) as notHelpfulIrrelevantSources
	, sum(notHelpfulOpinionSpeculation) as notHelpfulOpinionSpeculation
	, sum(notHelpfulNoteNotNeeded) as notHelpfulNoteNotNeeded
	, trim(group_concat(distinct ratedOnTweetId||' ')) as ratedOnTweetIds
from ratings
group by 1;


drop table sampleDataSet;
create table sampleDataSet as 
with topNotes as (
	select noteId from noteStats order by cnt desc limit 10000
)

select ratings.* from topNotes JOIN ratings using (noteId)
order by raterParticipantId
limit 100000;


select count(distinct noteId) from sampleDataSet;
select count(distinct raterParticipantId) from sampleDataSet;


select
	noteId,
	noteAuthorParticipantId,
	createdAtMillis,
	timestampMillisOfFirstNonNMRStatus,

create view currentStatus as
select  
noteId
, noteAuthorParticipantId
, createdAtMillis
, timestampMillisOfFirstNonNMRStatus
, firstNonNMRStatus
, max(timestampMillisOfCurrentStatus) AS timestampMillisOfLatestStatus
, currentStatus
, timestampMillisOfLatestNonNMRStatus
, mostRecentNonNMRStatus
, timestampMillisOfStatusLock
, lockedStatus
, timestampMillisOfRetroLock
, currentCoreStatus
, currentExpansionStatus
, currentGroupStatus
, currentDecidedBy
, currentModelingGroup
from noteStatusHistory
group by noteId;




drop table sampleDataSet2;
create table sampleDataSet2 as 
with topNotes as (
	select noteId from noteStats order by cnt desc limit 1000
)
, topUsers as (
	select raterParticipantId, count(*) as noteOverlapCount from ratings where noteId in (select noteId from topNotes)
	group by 1
	order by noteOverlapCount desc
	limit 1000
)
select ratings.* from topNotes JOIN ratings using (noteId) join topUsers using (raterParticipantId)
order by raterParticipantId
limit 100000;


select count(distinct noteId) from sampleDataSet2;
select count(distinct raterParticipantId) from sampleDataSet2;



drop table if exists sampleDataSet3;
create table sampleDataSet3 as 
with topNotes as (
	select noteId from noteStats order by cnt desc limit 10000
)
, topUsers as (
	select raterParticipantId, count(*) as noteOverlapCount from ratings where noteId in (select noteId from topNotes)
	group by 1
	order by noteOverlapCount desc
	limit 10000
)
select ratings.* from topNotes JOIN ratings using (noteId) join topUsers using (raterParticipantId)
order by raterParticipantId
limit 1000000;


