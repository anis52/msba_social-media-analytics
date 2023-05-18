create table audience_2 select distinct imdb_id,title from imdb_movie_overview;
alter table audience_2 add column open_week int(11) default 0 after title;

#insert open weel, last week, open week of promotion and last week of promotio
update audience_2 set open_week=(select min(week) from boxoffice where boxoffice.imdb_id=audience_2.imdb_id group by imdb_id);
alter table audience_2 add column last_week int(11) default 0 after open_week;
update audience_2 set last_week=(select max(week) from boxoffice where boxoffice.imdb_id=audience_2.imdb_id group by imdb_id);
alter table audience_2 add column open_pro_week int(11) default 0 after last_week;   #open promotion week, the week when the company start marketing promotion in fb
update audience_2 set open_pro_week=(select min(week) from fbposts where fbposts.imdb_id=audience_2.imdb_id group by imdb_id);
alter table audience_2 add column last_pro_week int(11) default 0 after open_pro_week;   #last promotion week, the week when the company finish marketing promotion in fb
update audience_2 set last_pro_week=(select max(week) from fbposts where fbposts.imdb_id=audience_2.imdb_id group by imdb_id);

#t1:before release period; t2:during release period; t3:after release period
alter table audience_2 add column num_post_t1 int(11) default 0 after last_pro_week;   #number of posts before release period
update audience_2 set num_post_t1=(select ifnull(count(*),0) from fbposts where fbposts.imdb_id=audience_2.imdb_id and fbposts.`week` <audience_2.open_week group by imdb_id);
alter table audience_2 add column num_post_t2 int(11) default 0 after num_post_t1;   #number of posts during release period
update audience_2 set num_post_t2=(select ifnull(count(*),0) from fbposts where fbposts.imdb_id=audience_2.imdb_id and audience_2.open_week<=fbposts.week and fbposts.week<=audience_2.last_week group by imdb_id);
alter table audience_2 add column num_post_t3 int(11) default 0 after num_post_t2;   #number of posts after release period
update audience_2 set num_post_t3=(select ifnull(count(*),0) from fbposts where fbposts.imdb_id=audience_2.imdb_id and audience_2.last_week<fbposts.week group by imdb_id);
alter table audience_2 add column num_post int(11) default 0 after num_post_t3;   #total number of posts
update audience_2 set num_post=(select ifnull(count(*),0) from fbposts where fbposts.imdb_id=audience_2.imdb_id group by imdb_id);

#average number of likes per post in different time period
alter table audience_2 add column avg_like_cnt_t1 float(11) default 0,
add column avg_like_cnt_t2 float(11) default 0,
add column avg_like_cnt_t3 float(11) default 0,
add column avg_like_cnt float(11) default 0;
update audience_2 set avg_like_cnt_t1=(select avg(likesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and fbposts.week<audience_2.open_week group by imdb_id);
update audience_2 set avg_like_cnt_t2=(select avg(likesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and audience_2.open_week<=fbposts.week and fbposts.week<=audience_2.last_week group by imdb_id);
update audience_2 set avg_like_cnt_t3=(select avg(likesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and fbposts.week<audience_2.last_week group by imdb_id);
update audience_2 set avg_like_cnt=(select avg(likesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id group by imdb_id);


#average number of comments per post in different time period
alter table audience_2  add column avg_com_cnt_t1 float(11) default 0,
add column avg_com_cnt_t2 float(11) default 0,
add column avg_com_cnt_t3 float(11) default 0,
add column avg_com_cnt float(11) default 0;;
update audience_2 set avg_com_cnt_t1=(select avg(commentsCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and fbposts.week<audience_2.open_week group by imdb_id);
update audience_2 set avg_com_cnt_t2=(select avg(commentsCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and audience_2.open_week<=fbposts.week and fbposts.week<=audience_2.last_week group by imdb_id);
update audience_2 set avg_com_cnt_t3=(select avg(commentsCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and fbposts.week<audience_2.last_week group by imdb_id);
update audience_2 set avg_com_cnt=(select avg(commentsCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id group by imdb_id);


#average number of shares per post in different time period
alter table audience_2  add column avg_share_cnt_t1 float(11) default 0,
add column avg_share_cnt_t2 float(11) default 0,
add column avg_share_cnt_t3 float(11) default 0,
add column avg_share_cnt float(11) default 0;;
update audience_2 set avg_share_cnt_t1=(select avg(sharesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and fbposts.week<audience_2.open_week group by imdb_id);
update audience_2 set avg_share_cnt_t2=(select avg(sharesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and audience_2.open_week<=fbposts.week and fbposts.week<=audience_2.last_week group by imdb_id);
update audience_2 set avg_share_cnt_t3=(select avg(sharesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id and fbposts.week<audience_2.last_week group by imdb_id);
update audience_2 set avg_share_cnt=(select avg(sharesCount) from fbposts where fbposts.imdb_id=audience_2.imdb_id group by imdb_id);


#star rating;
alter table audience_2 add column avg_star_rating_t1 float(11) default 0,
add column avg_star_rating_t2 float(11) default 0,
add column avg_star_rating_t3 float(11) default 0,
add column avg_star_rating float(11) default 0;
update audience_2 set avg_star_rating_t1=(select avg(star_rating) from fandango_review f where f.imdb_id=audience_2.imdb_id and f.week<audience_2.open_week group by imdb_id);
update audience_2 set avg_star_rating_t2=(select avg(star_rating) from fandango_review f where f.imdb_id=audience_2.imdb_id and audience_2.open_week<=f.week and f.week<=audience_2.last_week group by imdb_id);
update audience_2 set avg_star_rating_t3=(select avg(star_rating) from fandango_review f where f.imdb_id=audience_2.imdb_id and audience_2.last_week<f.week  group by imdb_id);
update audience_2 set avg_star_rating=(select avg(star_rating) from fandango_review f where f.imdb_id=audience_2.imdb_id group by imdb_id);
alter table audience_2 drop column avg_star_rating_t1,
drop column avg_star_rating_t2,
drop column avg_star_rating_t3;

#change null to 0;
update audience_2 a set num_post_t1=0 where num_post_t1 is null;
update audience_2 a set num_post_t2=0 where num_post_t2 is null;
update audience_2 a set num_post_t3=0 where num_post_t3 is null;
update audience_2 a set num_post=0 where num_post is null;

update audience_2 set avg_like_cnt_t1=0 where avg_like_cnt_t1 is null;
update audience_2 set avg_like_cnt_t2=0 where avg_like_cnt_t2 is null;
update audience_2 set avg_like_cnt_t3=0 where avg_like_cnt_t3 is null;
update audience_2 set avg_like_cnt=0 where avg_like_cnt is null;

update audience_2 set avg_com_cnt_t1=0 where avg_com_cnt_t1 is null;
update audience_2 set avg_com_cnt_t2=0 where avg_com_cnt_t2 is null;
update audience_2 set avg_com_cnt_t3=0 where avg_com_cnt_t3 is null;
update audience_2 set avg_com_cnt=0 where avg_com_cnt is null;

update audience_2 set avg_share_cnt_t1=0 where avg_share_cnt_t1 is null;
update audience_2 set avg_share_cnt_t2=0 where avg_share_cnt_t2 is null;
update audience_2 set avg_share_cnt_t3=0 where avg_share_cnt_t3 is null;
update audience_2 set avg_share_cnt=0 where avg_share_cnt is null;
