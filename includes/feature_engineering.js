function buildFeatures(source_table) {

  return `

WITH sess_duration AS (

  SELECT
    user_pseudo_id,
    ga_session_id,

    (MAX(event_timestamp) - MIN(event_timestamp)) / 1000000
      AS session_duration_sec

  FROM ${source_table}

  WHERE ga_session_id IS NOT NULL

  GROUP BY
    user_pseudo_id,
    ga_session_id

),

session_duration AS (

SELECT
  user_pseudo_id,

  COUNT(*) AS total_sessions,

  CAST(ROUND(AVG(session_duration_sec), 0) AS INT64)
    AS session_duration_avg,

  MAX(session_duration_sec)
    AS session_duration_max,

  CAST(ROUND(SUM(session_duration_sec),0) AS INT64)
    AS total_session_time

FROM sess_duration

GROUP BY user_pseudo_id

),

seen_products AS (

  SELECT
    user_pseudo_id,

    ARRAY_AGG(DISTINCT product ORDER BY product)
      AS id_seen_products,

    COUNT(*) AS num_products

  FROM (

    SELECT DISTINCT
      user_pseudo_id,
      SAFE_CAST(item_id AS INT64) AS product

    FROM ${source_table},
    UNNEST(item_ids) AS item_id

    WHERE SAFE_CAST(item_id AS INT64) IS NOT NULL

  )

  GROUP BY user_pseudo_id

),

campain AS (
  SELECT
    user_pseudo_id,

    ARRAY_AGG(DISTINCT campaign_ids IGNORE NULLS ORDER BY campaign_ids) AS id_campaigns,
    COUNT(DISTINCT campaign_ids) AS num_campaigns,

  FROM (
    SELECT
      user_pseudo_id,
      campaign_ids
    FROM ${source_table}
  )
  GROUP BY user_pseudo_id
),

device as (
  SELECT
    user_pseudo_id,

    ARRAY_AGG(DISTINCT device_used IGNORE NULLS ORDER BY device_used) AS used_devices,
    COUNT(DISTINCT device_used) AS num_devices

  FROM (
    SELECT
      user_pseudo_id,
      device_used,
    FROM ${source_table}
  )
  GROUP BY user_pseudo_id
),

canal as (
  SELECT
    user_pseudo_id,

    ARRAY_AGG(DISTINCT page_referrer IGNORE NULLS ORDER BY page_referrer) AS canal_id,
    COUNT(DISTINCT page_referrer) AS num_canals

  FROM (
    SELECT
      user_pseudo_id,
      page_referrer,
    FROM ${source_table}
  )
  GROUP BY user_pseudo_id
),

min_first_visit AS (
  SELECT
    user_pseudo_id,
    MIN(event_timestamp) AS first_visit_ts,
  FROM ${source_table}
  GROUP BY user_pseudo_id
),

features AS (
  SELECT
    user_pseudo_id,
    COUNT(*) AS total_events,
    COUNTIF(event_name = 'purchase') AS total_purchase,
    COUNTIF(event_name = 'page_view') AS total_pageviews,
    SAFE_DIVIDE(COUNT(*), COUNT(DISTINCT ga_session_id)) AS events_per_session,

    COUNTIF(event_name = 'scroll') AS total_scrolls,
    COUNTIF(event_name = 'user_engagement') AS total_user_engagement,

    COUNTIF(event_name = 'view_item') AS total_view_item,
    COUNTIF(event_name = 'select_item') AS total_select_item,
    COUNTIF(event_name = 'add_to_cart') AS total_add_to_cart,
    COUNTIF(event_name = 'begin_checkout') AS total_begin_checkout,

    COUNTIF(event_name = 'form_start') AS total_form_start,
    MAX(IF(event_name = 'form_success', 1, 0)) AS form_success_flag,
    MAX(event_name = 'form_success_unbounce') AS form_success_flag_unbounce_flag,
    MAX(IF(event_name = 'generate_lead', 1, 0)) AS generate_lead_flag,
    COUNTIF(event_name = 'open_form') AS total_open_form,
    MAX(event_name = 'book_call') AS book_call_flag,
    MAX(event_name = 'call_to_phone') AS call_to_phone_flag,
    COUNTIF(event_name = 'video_start') AS total_video_start,
    COUNTIF(event_name = 'video_progress') AS total_video_progress,
    MAX(IF(event_name = 'video_complete', 1, 0)) AS video_complete_flag,

    MAX(event_name = 'open_chat') AS chat_open_flag,
    MAX(event_name = 'chat_start') AS chat_start_flag,
    MAX(event_name = 'chat_success') AS chat_success_flag,
    COUNTIF(event_name = 'click_popup_banner') AS total_popup_clicks,
    COUNTIF(event_name = 'click') AS total_clicks,
    COUNTIF(event_name = 'click_slide') AS total_click_slide,

    -- CONTENT
    COUNTIF(event_name = 'file_download') AS total_file_download,
    COUNTIF(event_name = 'view_content') AS total_view_content,
    COUNTIF(event_name = 'view_item_list') AS total_view_item_list,
    COUNTIF(event_name = 'view_popup_banner') AS total_view_popup_banner,

    COUNTIF(event_name = 'search') AS total_search,
    COUNTIF(event_name = 'session_start') AS total_session_start,
    COUNTIF(event_name = 'first_visit') AS total_first_visit

  FROM ${source_table} 
  GROUP BY user_pseudo_id
)

SELECT 
f.*,
s.total_sessions,
s.session_duration_avg,
s.session_duration_max,
s.total_session_time,
TO_JSON_STRING(i.id_seen_products) as products,
i.num_products,
TO_JSON_STRING(c.id_campaigns) as campaings,
c.num_campaigns,
TO_JSON_STRING(d.used_devices) as used_devices,
d.num_devices,
TO_JSON_STRING(cn.canal_id) as used_canals,
cn.num_canals,
fv.first_visit_ts

FROM features f

LEFT JOIN session_duration s
USING(user_pseudo_id)

LEFT JOIN seen_products i
USING(user_pseudo_id)

LEFT JOIN campain c
USING(user_pseudo_id)

LEFT JOIN device d
USING(user_pseudo_id)

LEFT JOIN canal cn
USING(user_pseudo_id)

LEFT JOIN min_first_visit fv
USING(user_pseudo_id)

`;

}

module.exports = {
  buildFeatures
};