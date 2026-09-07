import { StudioNav } from '@/components/studio-nav';
import Link from 'next/link';
export default function Methods() {
  return (
    <main className="document-shell">
      <StudioNav active="/method" />
      <header className="document-heading">
        <span className="eyebrow">04 / Methods & limits</span>
        <h1>Make the evidence inspectable.</h1>
        <p>
          A working pipeline is not the same as a validated scientific model.
          Use this studio for investigation and review, not unattended
          operational decisions.
        </p>
      </header>
      <section className="document-panel method-table">
        <h2>Three evidence sets, distinct methods</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Workflow</th>
                <th>Current method</th>
                <th>What it establishes</th>
                <th>What remains unvalidated</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Single scene</td>
                <td>
                  Qwen3-VL + BigEarthNet adapter; instruction-model report. SAM2
                  proposal-refined masks when the quality notebook is running.
                  NDWI only with explicit green/NIR bands.
                </td>
                <td>
                  Visual descriptions and candidate boundaries on the source
                  grid.
                </td>
                <td>
                  Mask class correctness, fine species, complete object
                  inventory and geographic transfer.
                </td>
              </tr>
              <tr>
                <td>Two dates</td>
                <td>
                  Local normalized selected-band difference; shared-valid pixel
                  masks by default. Optional trained CDVQA/SECOND expert uses
                  a learned paired encoder and supervised change head. Target
                  extent requests chain two segmentations and a separate measurement.
                </td>
                <td>Where values differ above a declared threshold.</td>
                <td>
                  Semantic change, registration quality, cloud-free validity and
                  physical cause.
                </td>
              </tr>
              <tr>
                <td>Optical + SAR</td>
                <td>
                  Local color/texture + relative backscatter proxy, masks on the
                  optical reference by default. A separately trained TerraMind
                  expert can provide learned Sentinel-1/2 scene labels—not precision masks.
                </td>
                <td>Inspectible complementary signal candidates.</td>
                <td>
                  Learned fusion accuracy, absolute radar calibration and class
                  correctness.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
      <div className="document-grid">
        <section className="document-panel">
          <h2>Plans you can audit</h2>
          <p>
            The Kaggle instruction model can propose multiple objectives. The
            local controller validates them and builds a bounded dependency plan;
            it never executes model-written code. Trace provenance explicitly
            identifies learned planning, user-selected tasks or deterministic fallback.
          </p>
          <p>
            A reservoir comparison can segment each date, compare scenes, then
            measure candidate loss and gain. Missing masks or mismatched grids
            withhold the measurement. Surface extent is not water level.
          </p>
        </section>
        <section className="document-panel">
          <h2>Sensor identity matters</h2>
          <p>
            Embedded product headers and band descriptions are preserved.
            Cartosat multispectral B2 is green and B4 is NIR; Sentinel numbering
            is different. Unknown numeric bands remain unknown.
          </p>
          <p>
            RISAT/EOS-04 RH/RV and HH/HV remain distinct polarizations. They
            cannot be relabelled as the VV/VH inputs of a Sentinel-trained expert.
            Pixel spacing does not establish native resolution or calibration.
          </p>
        </section>
        <section className="document-panel">
          <h2>Image requirements</h2>
          <ul>
            <li>
              Exploration: RGB/grayscale JPG, PNG, WebP and TIFF; maximum 50 MB
              per file. Display images are limited to 40 megapixels.
            </li>
            <li>
              Convert palette, CMYK, alpha-channel, rotated-EXIF or animated
              images into an upright RGB image first.
            </li>
            <li>
              SIH strict: GeoTIFF/TIFF; prescribed benchmark PNG/JPEG imports
              remain separate.
            </li>
            <li>
              Unreferenced temporal exports need identical dimensions, bands and
              sample types, plus an explicit pixel-alignment declaration. No
              geographic area is reported.
            </li>
            <li>
              Fusion needs genuine co-registered optical and SAR rasters. Two
              ordinary photos cannot substitute for two sensor modalities.
            </li>
          </ul>
        </section>
        <section className="document-panel">
          <h2>For forestry and reporting</h2>
          <ul>
            <li>
              Describe canopy pattern, visible fragmentation and bare surfaces
              as observations, not species or legal classifications.
            </li>
            <li>
              Review all boundaries against original pixels. A plausible mask
              can still include shadows or miss small water bodies.
            </li>
            <li>
              Verify the acquisition date, location, sensor and licenses.
              Separate camera coordinates from image footprint.
            </li>
            <li>
              Keep competing explanations visible. Cloud, season, illumination,
              viewing angle and alignment can mimic physical change.
            </li>
            <li>
              Use case JSON and PDF exports to preserve source hashes, method
              versions, warnings and measured mask coverage.
            </li>
          </ul>
        </section>
        <section className="document-panel">
          <h2>Accuracy must be measured</h2>
          <p>
            There is no calibrated correctness percentage for arbitrary uploaded
            scenes. Coverage is the fraction of selected pixels, not confidence.
          </p>
          <p>
            Evaluate held-out local imagery using IoU, Dice, precision/recall,
            boundary error and abstention rate. Include no-feature scenes and
            difficult cloud/shadow examples. Compare any new model against this
            baseline before promoting it.
          </p>
        </section>
        <section className="document-panel">
          <h2>Local and free, with limits</h2>
          <p>
            The website, casebook, reports and pair baselines run on your
            machine. Single-scene VLM inference needs your active Kaggle GPU
            plus protected ngrok session.
          </p>
          <p>
            Nothing here provisions paid hosting. Free GPU/tunnel quotas and
            session lifetimes still apply. An expired Kaggle session means the
            single-scene model is unavailable—not a reason to return a fake
            answer.
          </p>
          <Link href="/archive">Find historical satellite and weather context →</Link>
        </section>
      </div>
    </main>
  );
}
