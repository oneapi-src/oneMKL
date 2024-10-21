# oneAPI Math Kernel Library (oneMKL) Interfaces Design Documents / RFCs

This branch contains design documents for oneMKL Interfaces project-wide changes. The purpose of Request for Comments (RFC) process is to communicate all major changes in the project prior the actual implementation and document the decisions in one place.

All design documents (RFCs) that are approved for implementation should be merged to this branch.

## Document Style

* Every design documents should be added as markdown document
`rfcs/<YYYMMDD>-descriptive-but-short-proposal-name/README.md`.
    * [Optional] For very domain specific documents location could be
`rfcs/<domain>/<YYYMMDD>-descriptive-but-short-proposal-name/README.md`
* Additional to `README.md` the design document directory can contain any other
supporting materials: images, formulas, sub-proposal, etc.
* The recommended document structure:
[RFC template](rfcs/template.md).
* Recommended width of the raw text is 80-100 symbols,
long lines make it hard to read the document in the raw format


## RFC Ratification Process

1. Add new design document as a PR to this repository
    * Please add a link to preview document in the PR description,
e.g. link for this README in your fork will be
        ```
        https://github.com/<USERNAME>/oneMKL/blob/rfcs/README.md
        ```
2. Assign all affected [teams](https://github.com/oneapi-src/oneMKL/blob/develop/README.md#contributing) and individual
contributors as reviewers to the PR.
3. Add `RFC` label to the PR to trigger slack notification in [#onemkl](https://uxlfoundation.slack.com/archives/onemkl) channel.
4. Organize offline review or/and bring the RFC to [UXL Foundation Math SIG forum](https://lists.uxlfoundation.org/g/Math-SIG), [UXL Foundation Open Source Working Group](https://lists.uxlfoundation.org/g/open-source-wg), or any other related forums in order to collect feedback.
    * It's recommended to keep all feedback as part of PR review, so it also
will be documented in one place
5. If changes affect API defined by [oneMKL specification](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/onemkl/source/) the related part of the design document must be converted to oneAPI specification RFC (as a new [issue](https://github.com/uxlfoundation/oneAPI-spec/issues) with \[RFC\] tag), reviewed by [UXL Foundation Math SIG forum](https://lists.uxlfoundation.org/g/Math-SIG), and contributed to [oneAPI specification](https://github.com/uxlfoundation/oneAPI-spec), and only after it the proposed changes can be implemented in this project.
6. Merge PR when it has all required approvals
    * It's recommended to add PR number to the commit message, so it will be easy
to find the design discussion
    * It's recommended to update the preview document link in the PR to the merged
one because initial link to the local fork/branch will stop working after local branch removal,
e.g. link for this README will be 
        ```
        https://github.com/oneapi-src/oneMKL/blob/rfcs/README.md
        ```

