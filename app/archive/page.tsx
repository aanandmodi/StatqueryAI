'use client';
import { SyntheticEvent, useState } from 'react';
import { StudioNav } from '@/components/studio-nav';
import { jsonRequest } from '@/lib/api-client';

type History = {
  source: string;
  warnings: string[];
  items: {
    id: string;
    acquired_at: string;
    cloud_cover_percent: number | null;
    platform: string;
    source_url: string;
    assets: { [key: string]: string };
  }[];
};
type Weather = {
  source: string;
  source_url: string;
  warnings: string[];
  parameters: { [key: string]: { units: string } };
  days: ({ date: string } & { [key: string]: string | number | null })[];
};
function saveEvidence(data: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }),
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export default function Archive() {
  const [history, setHistory] = useState<History | null>(null);
  const [weather, setWeather] = useState<Weather | null>(null);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [weatherBusy, setWeatherBusy] = useState(false);
  const [historyError, setHistoryError] = useState('');
  const [weatherError, setWeatherError] = useState('');
  async function search(
    event: SyntheticEvent<HTMLFormElement>,
    kind: 'history' | 'weather',
  ) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const val = (key: string) => {
      const value = form.get(key);
      return typeof value === 'string' ? value : '';
    };
    const payload =
      kind === 'history'
        ? {
            bbox: ['west', 'south', 'east', 'north'].map((key) =>
              Number(val(key)),
            ),
            acquisition_date: val('acquisition'),
            start_date: val('start'),
            end_date: val('end'),
            max_cloud_cover: Number(val('cloud')),
            footprint_confirmed: form.has('confirmed'),
            limit: 8,
          }
        : {
            latitude: Number(val('latitude')),
            longitude: Number(val('longitude')),
            start_date: val('start'),
            end_date: val('end'),
          };
    if (kind === 'history') {
      setHistoryBusy(true);
      setHistoryError('');
      setHistory(null);
    } else {
      setWeatherBusy(true);
      setWeatherError('');
      setWeather(null);
    }
    try {
      const result = await jsonRequest<History | Weather>(
        `/api/satquery/evidence/${kind}/search`,
        {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload),
        },
      );
      if (kind === 'history') setHistory(result as History);
      else setWeather(result as Weather);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Evidence source unavailable';
      if (kind === 'history') setHistoryError(message);
      else setWeatherError(message);
    } finally {
      if (kind === 'history') setHistoryBusy(false);
      else setWeatherBusy(false);
    }
  }
  return (
    <main className="document-shell">
      <StudioNav active="/archive" />
      <header className="document-heading">
        <span className="eyebrow">03 / Historical evidence</span>
        <h1>One image is a starting point.</h1>
        <p>
          Find earlier satellite acquisitions and regional weather context. No
          API key or paid imagery service. You must know the location and date;
          SatQuery does not guess them from a photo.
        </p>
      </header>
      <div className="document-grid">
        <section className="document-panel">
          <h2>Find a before scene</h2>
          <p>
            Sentinel-2 L2A catalog discovery. Enter the mapped footprint—not
            only the camera’s GPS point. Select dates before your image was
            acquired.
          </p>
          <form onSubmit={(e) => search(e, 'history')}>
            <div className="form-grid">
              {[
                ['west', 'West longitude', '85.2'],
                ['south', 'South latitude', '27.5'],
                ['east', 'East longitude', '85.5'],
                ['north', 'North latitude', '27.8'],
              ].map(([name, label, placeholder]) => (
                <label key={name}>
                  {label}
                  <input
                    name={name}
                    type="number"
                    step="any"
                    required
                    placeholder={placeholder}
                  />
                </label>
              ))}
              <label>
                Your image acquisition date
                <input type="date" name="acquisition" required />
              </label>
              <label>
                Maximum scene cloud %
                <input
                  type="number"
                  name="cloud"
                  min="0"
                  max="100"
                  defaultValue="30"
                  required
                />
              </label>
              <label>
                Search from
                <input type="date" name="start" required />
              </label>
              <label>
                Search through
                <input type="date" name="end" required />
              </label>
              <label className="inline-check">
                <input type="checkbox" name="confirmed" required />I confirm
                this box represents the mapped area, and the acquisition date is
                known.
              </label>
            </div>
            <button disabled={historyBusy}>
              {historyBusy
                ? 'Searching catalog…'
                : 'Search earlier acquisitions'}
            </button>
          </form>
          <p className="small-note">
            At most one year and a 2° × 2° region per request. Location and
            dates are sent to Earth Search only when you search. Images are not
            uploaded.
          </p>
          {historyError && (
            <div role="alert" className="error-banner">
              {historyError}
            </div>
          )}
          {history && (
            <div className="context-results">
              <h3>{history.items.length} candidate acquisitions</h3>
              {history.warnings.map((w) => (
                <p key={w}>{w}</p>
              ))}
              {!history.items.length && (
                <p>
                  No scenes matched. Widen the date window or cautiously relax
                  the cloud limit.
                </p>
              )}
              {history.items.map((item) => (
                <article key={item.id}>
                  <span className="eyebrow">Not comparison-ready</span>
                  <h3>{item.id}</h3>
                  <p>
                    {item.acquired_at} · {item.platform} · scene cloud{' '}
                    {item.cloud_cover_percent ?? 'unknown'}%
                  </p>
                  <a href={item.source_url} target="_blank" rel="noreferrer">
                    Original catalog record ↗
                  </a>
                  <details>
                    <summary>Available public image assets</summary>
                    <p className="small-note">
                      Full-scene COG files can be large. These links do not
                      automatically import or align imagery.
                    </p>
                    <div className="source-links">
                      {Object.entries(item.assets).map(([name, url]) => (
                        <a
                          key={name}
                          href={url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {name} ↗
                        </a>
                      ))}
                    </div>
                  </details>
                </article>
              ))}
              <button
                onClick={() =>
                  saveEvidence(history, 'satquery-historical-evidence.json')
                }
              >
                Export search provenance
              </button>
            </div>
          )}
        </section>
        <section className="document-panel">
          <h2>Check regional weather context</h2>
          <p>
            NASA POWER daily reanalysis. Useful for investigating an event, not
            proof of weather at an exact pixel or proof of flood causation.
          </p>
          <form onSubmit={(e) => search(e, 'weather')}>
            <div className="form-grid">
              <label>
                Latitude
                <input
                  name="latitude"
                  type="number"
                  min="-90"
                  max="90"
                  step="any"
                  required
                  placeholder="27.717"
                />
              </label>
              <label>
                Longitude
                <input
                  name="longitude"
                  type="number"
                  min="-180"
                  max="180"
                  step="any"
                  required
                  placeholder="85.324"
                />
              </label>
              <label>
                From
                <input name="start" type="date" required />
              </label>
              <label>
                Through
                <input name="end" type="date" required />
              </label>
            </div>
            <button disabled={weatherBusy}>
              {weatherBusy
                ? 'Retrieving context…'
                : 'Retrieve weather evidence'}
            </button>
          </form>
          <p className="small-note">
            At most 31 days. Coordinates and dates are sent to NASA only when
            requested. Coarse regional grid; valleys and local conditions may
            differ.
          </p>
          {weatherError && (
            <div role="alert" className="error-banner">
              {weatherError}
            </div>
          )}
          {weather && (
            <div className="context-results">
              <h3>{weather.source}</h3>
              {weather.warnings.map((w) => (
                <p key={w}>{w}</p>
              ))}
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Date UTC</th>
                      {['T2M', 'PRECTOTCORR', 'RH2M', 'WS10M'].map((key, i) => (
                        <th key={key}>
                          {
                            [
                              'Temperature',
                              'Precipitation',
                              'Humidity',
                              'Wind',
                            ][i]
                          }{' '}
                          (
                          {weather.parameters[key]?.units ?? 'unit unavailable'}
                          )
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {weather.days.map((day) => (
                      <tr key={day.date}>
                        <td>{day.date}</td>
                        {['T2M', 'PRECTOTCORR', 'RH2M', 'WS10M'].map((key) => (
                          <td key={key}>{day[key] ?? 'Missing'}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="toolbar">
                <a href={weather.source_url} target="_blank" rel="noreferrer">
                  NASA methodology ↗
                </a>
                <button
                  onClick={() =>
                    saveEvidence(weather, 'satquery-weather-context.json')
                  }
                >
                  Export sourced context
                </button>
              </div>
            </div>
          )}
        </section>
      </div>
      <section className="document-panel" style={{ marginTop: 24 }}>
        <h2>What happens after discovery?</h2>
        <ol>
          <li>
            Choose a cloud-free, seasonally appropriate reference acquisition
            that covers the same area.
          </li>
          <li>
            Crop and co-register compatible sensor rasters; verify CRS,
            resolution, band meanings, date and cloud/shadow quality.
          </li>
          <li>
            Upload the verified before/after pair in Investigation. A phone
            photograph and a satellite scene cannot be subtracted as matching
            pixels.
          </li>
          <li>
            Use weather and authoritative event records to assess competing
            explanations. Discovery does not automatically establish change or
            its cause.
          </li>
        </ol>
        <p>
          Catalog sources:{' '}
          <a href="https://element84.com/earth-search/">
            Element 84 Earth Search
          </a>{' '}
          ·{' '}
          <a href="https://registry.opendata.aws/sentinel-2-l2a-cogs/">
            Sentinel open data
          </a>
          . Automatic crop/import/registration is not implemented in this
          release.
        </p>
      </section>
    </main>
  );
}
