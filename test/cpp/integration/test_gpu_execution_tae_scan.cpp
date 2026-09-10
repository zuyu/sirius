/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// End-to-end GPU coverage for the TAE scanner. These use the scanner's checked-in
// fixtures and disable DuckDB fallback so a missing planner admission or a runtime
// incompatibility cannot silently pass on CPU.

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <filesystem>
#include <string>

namespace {

std::string tae_manifest(const std::string& name)
{
#ifdef SIRIUS_PROJECT_ROOT
  auto const path =
    std::filesystem::path(SIRIUS_PROJECT_ROOT) / "tae-scanner" / "test" / "data" / name;
#else
  auto const path =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
    "tae-scanner" / "test" / "data" / name;
#endif
  REQUIRE(std::filesystem::exists(path));
  return path.string();
}

std::string tae_scan(const std::string& manifest) { return "tae_scan('" + manifest + "')"; }

class TaeScanGpuFixture : public sirius::test::GpuExecutionFixture {
 public:
  TaeScanGpuFixture() { run_ok("SET enable_duckdb_fallback = false;"); }

  ~TaeScanGpuFixture() { con->Query("SET enable_duckdb_fallback = true;"); }
};

}  // namespace

TEST_CASE_METHOD(TaeScanGpuFixture,
                 "gpu_execution executes tae_scan fixtures on GPU without DuckDB fallback",
                 "[integration][gpu_execution][tae_scan]")
{
  SECTION("basic projection and aggregation")
  {
    auto const scan = tae_scan(tae_manifest("manifest.json"));
    compare_gpu_vs_cpu("SELECT col_int, col_str FROM " + scan + " ORDER BY col_int");
    compare_gpu_vs_cpu("SELECT col_str, col_int FROM " + scan + " ORDER BY col_int");
    compare_gpu_vs_cpu("SELECT count(*), min(col_int), max(col_int) FROM " + scan);
  }

  SECTION("LZ4 data with a filter on a non-projected column")
  {
    auto const scan = tae_scan(tae_manifest("manifest_lz4.json"));
    compare_gpu_vs_cpu("SELECT col_str FROM " + scan + " WHERE col_int > 50 ORDER BY col_str");
    compare_gpu_vs_cpu("SELECT col_str FROM " + scan + " WHERE col_int > 100000");
  }

  SECTION("nullable values and multi-file manifests")
  {
    auto const null_scan = tae_scan(tae_manifest("manifest_nulls.json"));
    compare_gpu_vs_cpu("SELECT col_int, col_str FROM " + null_scan +
                       " WHERE col_int IS NULL OR col_str IS NULL ORDER BY col_int NULLS FIRST");

    auto const multi_scan = tae_scan(tae_manifest("manifest_multifile.json"));
    compare_gpu_vs_cpu("SELECT col_int, col_str FROM " + multi_scan +
                       " WHERE col_int > 50 ORDER BY col_int");
  }

  SECTION("constant vectors across multiple blocks")
  {
    auto const scan = tae_scan(tae_manifest("manifest_constants.json"));
    compare_gpu_vs_cpu("SELECT col_int, col_str, col_dbl FROM " + scan +
                       " ORDER BY col_int, col_str");
  }

  SECTION("MatrixOne date and timestamp epochs")
  {
    auto const scan = tae_scan(tae_manifest("manifest_datetime.json"));
    compare_gpu_vs_cpu("SELECT col_date, col_ts, col_ref FROM " + scan + " ORDER BY col_ref");
  }
}
