# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# This file is included by DuckDB's build system. It specifies which extension
# to load

# The native TAE GPU ingestible shares the scanner's table-function bind data
# and zone-map helpers. Build the pinned sidecar extension with Sirius so
# `tae_scan` is registered before Sirius converts its physical plan.
duckdb_extension_load(
  tae_scanner SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/tae-scanner INCLUDE_DIR
  ${CMAKE_CURRENT_LIST_DIR}/tae-scanner/include)

# Extension from this repo
duckdb_extension_load(sirius SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} LOAD_TESTS
                      EXTENSION_VERSION dev)
